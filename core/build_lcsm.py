import geopandas as gpd
import pandas as pd
from rasterio.transform import from_origin
from rasterio.features import rasterize
import numpy as np
import logging
import json
import time
from pathlib import Path
from shapely import LineString
from shapely.ops import unary_union
from typing import List, Dict, Union
from concurrent.futures import ProcessPoolExecutor
from shapely.strtree import STRtree
from scipy.ndimage import generic_filter
from utils.timer import timeit
from utils.geodata_io import read_geodata
import psutil
import re
import gc
import sys
from multiprocessing import Process, Queue

logger = logging.getLogger(__name__)

def log_memory_usage():
    """记录当前内存使用情况"""
    mem = psutil.virtual_memory()
    logger.info(f"💾 内存使用: {mem.used / 1024**3:.2f}GB/{mem.total / 1024**3:.2f}GB (可用: {mem.available / 1024**3:.2f}GB)")

def normalize_column_name(name):
    """统一字段名格式（小写+去特殊字符）"""
    return re.sub(r'[^a-z0-9]', '', name.lower().strip())

def find_column_by_normalized(df, target):
    """通过标准化名称查找字段"""
    target_norm = normalize_column_name(target)
    for col in df.columns:
        if normalize_column_name(col) == target_norm:
            return col
    return None

@timeit("加载权重配置")
def load_weights(json_path: Union[str, Path] = "config/weight_config.json") -> Dict[str, float]:
    """
    加载LCSM权重配置
    """
    json_path = Path(json_path)
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 设置默认值（防止配置缺失）
        config.setdefault('lcsm_distance_range', [10, 300])
        config.setdefault('distance_decay', {'a': 1000, 'b': 0.5})
        config.setdefault('impact_decay_factor', 0.01)
        config.setdefault('lcsm_weights', {
            "transition_frequency": 0.5,
            "contact_density": 0.3,
            "mixture_index": 0.2
        })

        logger.info(f"✅ 加载LCSM配置: { {k: v for k, v in config.items() if k != 'laim_weights'} }")
        return config
    except Exception as e:
        logger.critical(f"❌ LCSM配置加载失败: {json_path} | 错误: {str(e)}")
        # 返回安全的默认配置
        return {
            'lcsm_weights': {"transition_frequency": 0.5, "contact_density": 0.3, "mixture_index": 0.2},
            'lcsm_distance_range': [10, 300],
            'distance_decay': {'a': 1000, 'b': 0.5},
            'impact_decay_factor': 0.01
        }

def validate_dataframe(df: pd.DataFrame, required_cols: List[str]) -> pd.DataFrame:
    """验证DataFrame是否包含所需字段，缺失则填充默认值0"""
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.warning(f"⚠️ 数据框缺失字段: {missing_cols}，自动填充默认值0")
        for col in missing_cols:
            df[col] = 0
    return df

# 计算地类转化频率 T(i,j)
def compute_transition_frequency(gdf_list: List[gpd.GeoDataFrame], resolution: int = 30) -> pd.DataFrame:
    """
    栅格化方法计算地类转化频率 T(i,j)
    使用向量化操作替代循环
    """
    # 验证输入数据有效性
    if len(gdf_list) < 2:
        logger.error("❌ 至少需要两个时间段的数据才能计算地类转化频率")
        return pd.DataFrame(columns=['class_a', 'class_b', 'T'])

    transition_counts = {}
    total_transitions = 0

    for i in range(len(gdf_list) - 1):
        gdf1, gdf2 = gdf_list[i], gdf_list[i + 1]
        bounds = gpd.GeoSeries([gdf1.unary_union, gdf2.unary_union]).unary_union.bounds
        transform = from_origin(bounds[0], bounds[3], resolution, resolution)
        width = int((bounds[2] - bounds[0]) / resolution) + 1
        height = int((bounds[3] - bounds[1]) / resolution) + 1

        # 栅格化（使用整型编码）
        raster1 = rasterize(
            [(geom, int(code)) for geom, code in zip(gdf1.geometry, gdf1.land_code)],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.int32  # 改为32位避免溢出
        )
        raster2 = rasterize(
            [(geom, int(code)) for geom, code in zip(gdf2.geometry, gdf2.land_code)],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.int32
        )

        # 向量化统计变化（增加空数组保护）
        valid_mask = (raster1 != 0) & (raster2 != 0) & (raster1 != raster2)
        from_classes = raster1[valid_mask]
        to_classes = raster2[valid_mask]

        # 检查数组是否为空
        if from_classes.size == 0 or to_classes.size == 0:
            logger.warning(f"⚠️ 时间段 {i} 到 {i + 1} 无有效变化数据")
            continue

        # 确保数组是1维的
        if from_classes.ndim > 1:
            from_classes = from_classes.flatten()
        if to_classes.ndim > 1:
            to_classes = to_classes.flatten()

        # 使用np.unique快速统计变化对
        stacked = np.column_stack((from_classes, to_classes))
        pairs, counts = np.unique(stacked, axis=0, return_counts=True)

        for (from_class, to_class), count in zip(pairs, counts):
            key = (int(from_class), int(to_class))
            transition_counts[key] = transition_counts.get(key, 0) + count
            total_transitions += count

    # 处理无转化数据的情况
    if total_transitions == 0:
        logger.warning("⚠️ 所有时间段均无地类转化数据，使用默认频率0")
        # 获取所有地类代码
        all_codes = set()
        for gdf in gdf_list:
            all_codes.update(gdf['land_code'].unique())
        # 创建默认转化频率（同类为1，异类为0）
        t_matrix = []
        for i in all_codes:
            for j in all_codes:
                t_matrix.append({
                    'class_a': int(i),
                    'class_b': int(j),
                    'T': 1.0 if i == j else 0.0
                })
        return pd.DataFrame(t_matrix)

    # 创建转化频率矩阵
    t_matrix = [
        {'class_a': i, 'class_b': j, 'T': count / total_transitions}
        for (i, j), count in transition_counts.items()
    ]
    return validate_dataframe(pd.DataFrame(t_matrix), ['class_a', 'class_b', 'T'])

# 计算边界接触密度 B(i,j)
def compute_boundary_density(gdf_list: List[gpd.GeoDataFrame]) -> pd.DataFrame:
    """
    计算所有年份中地类对的平均边界接触密度 B(i,j)
    使用空间索引加速边界相交计算
    """
    results = {}

    # 获取所有地类代码
    all_codes = set()
    for gdf in gdf_list:
        all_codes.update(gdf['land_code'].unique())

    # 空数据检查
    if not all_codes:
        logger.error("❌ 无有效的地类数据")
        return pd.DataFrame(columns=['class_a', 'class_b', 'B'])

    for gdf in gdf_list:
        # 按地类分组
        grouped = gdf.groupby('land_code')
        class_codes = list(grouped.groups.keys())

        # 跳过空数据
        if not class_codes:
            continue

        # 使用unary_union合并几何
        class_polygons = {code: unary_union(group.geometry) for code, group in grouped}

        # 计算地类间最小距离（替代边界接触密度）
        for i, code_i in enumerate(class_codes):
            geom_i = class_polygons[code_i]
            for j, code_j in enumerate(class_codes):
                if i == j:
                    continue
                geom_j = class_polygons[code_j]

                # 计算最小距离（距离越小接触密度越大）
                min_dist = geom_i.distance(geom_j)
                key = (code_i, code_j)
                # 使用距离的倒数作为接触密度（距离越小密度越大）
                contact_density = 1 / (min_dist + 1)  # +1避免除零
                results.setdefault(key, []).append(contact_density)

    # 处理无边界接触的情况
    if not results:
        logger.warning("⚠️ 无边界接触数据，使用默认值0")
        b_matrix = []
        for i in all_codes:
            for j in all_codes:
                if i != j:
                    b_matrix.append({
                        'class_a': int(i),
                        'class_b': int(j),
                        'B': 0.0
                    })
        return validate_dataframe(pd.DataFrame(b_matrix), ['class_a', 'class_b', 'B'])

    # 按地类对求平均值
    b_matrix = [
        {'class_a': int(i), 'class_b': int(j), 'B': np.mean(vals) if vals else 0}
        for (i, j), vals in results.items()
    ]
    return validate_dataframe(pd.DataFrame(b_matrix), ['class_a', 'class_b', 'B'])

# 计算混合程度指数 M(i,j)（滑动窗口共现频率）
def compute_mixing_index(gdf_list: List[gpd.GeoDataFrame], resolution: int = 30, window_size: int = 3) -> pd.DataFrame:
    """
    滑动窗口法计算混合程度指数 M(i,j)
    使用向量化窗口操作替代双重循环
    """
    mixing_counts = {}
    total_count = 0
    CHUNK_SIZE = 1000  # 分块大小

    # 获取所有地类代码
    all_codes = set()
    for gdf in gdf_list:
        all_codes.update(gdf['land_code'].unique())

    # 空数据检查
    if not all_codes:
        logger.error("❌ 无有效的地类数据")
        return pd.DataFrame(columns=['class_a', 'class_b', 'M'])

    for gdf in gdf_list:
        bounds = gdf.total_bounds
        transform = from_origin(bounds[0], bounds[3], resolution, resolution)
        width = int((bounds[2] - bounds[0]) / resolution) + 1
        height = int((bounds[3] - bounds[1]) / resolution) + 1

        # 栅格化（使用整型编码）
        raster = rasterize(
            [(geom, int(code)) for geom, code in zip(gdf.geometry, gdf.land_code)],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.int16
        )

        # 定义混合计数函数
        def count_mixing(window):
            center = window[window_size // 2, window_size // 2]
            if center == 0:
                return 0
            unique_vals = np.unique(window)
            # 移除0和中心值
            unique_vals = unique_vals[(unique_vals != 0) & (unique_vals != center)]
            return len(unique_vals)

        # 分块处理大型栅格
        for y in range(0, height, CHUNK_SIZE):
            y_end = min(y + CHUNK_SIZE, height)
            for x in range(0, width, CHUNK_SIZE):
                x_end = min(x + CHUNK_SIZE, width)

                # 仅处理包含有效数据的区块
                chunk = raster[y:y_end, x:x_end]
                if np.all(chunk == 0):
                    continue

                # 应用滑动窗口
                mixing_chunk = generic_filter(
                    chunk,
                    count_mixing,
                    size=(window_size, window_size),
                    mode='constant',
                    cval=0
                )

                # 统计当前区块
                for land_code in np.unique(chunk):
                    if land_code == 0:
                        continue
                    mask = (chunk == land_code)
                    count = np.sum(mixing_chunk[mask])
                    mixing_counts[land_code] = mixing_counts.get(land_code, 0) + count
                    total_count += count

                # 及时释放内存
                del mixing_chunk
                gc.collect()

    # 处理无混合数据的情况
    if total_count == 0:
        logger.warning("⚠️ 无混合数据，使用默认值0")
        m_matrix = []
        for i in all_codes:
            for j in all_codes:
                if i != j:
                    m_matrix.append({
                        'class_a': int(i),
                        'class_b': int(j),
                        'M': 0.0
                    })
        return validate_dataframe(pd.DataFrame(m_matrix), ['class_a', 'class_b', 'M'])

    # 转换为地类对形式
    total_m = sum(mixing_counts.values())
    m_matrix = []
    for code, count_val in mixing_counts.items():
        for other_code in mixing_counts.keys():
            if code != other_code:
                m_matrix.append({
                    'class_a': int(code),
                    'class_b': int(other_code),
                    'M': count_val / total_m
                })
    return validate_dataframe(pd.DataFrame(m_matrix), ['class_a', 'class_b', 'M'])

def save_lcsm_matrix(df: pd.DataFrame, value_col: str, output_path: Union[str, Path]) -> None:
    """
    保存为耦合强度矩阵格式（对角线置1）
    """
    output_path = Path(output_path)
    logger.info("📐 正在转换为耦合强度矩阵格式...")

    # 创建透视表
    pivot_df = df.pivot_table(index='class_a', columns='class_b', values=value_col, fill_value=0)

    # 确保所有地类都在行列中
    all_codes = sorted(set(pivot_df.index).union(pivot_df.columns))
    pivot_df = pivot_df.reindex(index=all_codes, columns=all_codes, fill_value=0)

    # 对角线置1（同类地物完全作用）
    np.fill_diagonal(pivot_df.values, 1.0)

    # 验证对角线
    diag_values = pivot_df.values.diagonal()
    logger.info(f"✅ 矩阵对角线验证: min={diag_values.min():.4f}, max={diag_values.max():.4f} (应全为1.0)")

    pivot_df.to_csv(output_path)
    logger.info(f"💾 已保存地类耦合强度矩阵至 {output_path}")

def _run_worker(func, args, queue):
    """多进程工作函数（模块级，确保可pickle）"""
    try:
        result = func(*args)
        queue.put(result)
    except Exception as e:
        queue.put(e)

def run_with_timeout(func, args, timeout=7200):
    """安全执行函数 - 模块级函数避免pickle问题"""
    q = Queue()
    p = Process(target=_run_worker, args=(func, args, q))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        raise TimeoutError(f"⌛ {func.__name__} 计算超时 ({timeout}s)")

    result = q.get()
    if isinstance(result, Exception):
        raise result
    return result

@timeit("构建地类耦合强度图谱 LCSM")
def build_lcsm(input_paths: List[Union[str, Path]], output_dir: Union[str, Path], resolution: int = 30) -> None:
    log_memory_usage()
    # 统一路径管理
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("📥 加载多期 features 图层...")

    def check_memory(min_avail_gb=2.0):
        mem = psutil.virtual_memory()
        avail_gb = mem.available / 1024 ** 3
        if avail_gb < min_avail_gb:
            logger.warning(f"🛑 可用内存低于{min_avail_gb}GB ({avail_gb:.2f}GB)，触发GC回收")
            gc.collect()
            new_mem = psutil.virtual_memory()
            logger.info(f"♻️ GC后可用内存: {new_mem.available / 1024 ** 3:.2f}GB")
        return mem.available

    # 增量加载多期数据
    gdf_list = []
    for path in input_paths:
        try:
            # 内存警戒线检查
            check_memory(1.5)

            gdf = read_geodata(str(path))

            # 字段名兼容性处理
            land_code_col = find_column_by_normalized(gdf, "land_code")
            if not land_code_col:
                logger.error(f"❌ 文件 {path} 缺少 land_code 字段，跳过处理")
                continue
            if land_code_col != "land_code":
                gdf = gdf.rename(columns={land_code_col: "land_code"})

            # 强制转换 land_code 为整数
            gdf['land_code'] = gdf['land_code'].astype(int)

            # 动态修复重心坐标字段
            centroid_x_col = find_column_by_normalized(gdf, "centroid_x")
            centroid_y_col = find_column_by_normalized(gdf, "centroid_y")
            if not centroid_x_col or not centroid_y_col:
                logger.warning(f"⚠️ 文件 {path} 缺少重心坐标字段，自动计算...")
                gdf['centroid_x'] = gdf.geometry.centroid.x
                gdf['centroid_y'] = gdf.geometry.centroid.y
            else:
                if centroid_x_col != "centroid_x":
                    gdf = gdf.rename(columns={centroid_x_col: "centroid_x"})
                if centroid_y_col != "centroid_y":
                    gdf = gdf.rename(columns={centroid_y_col: "centroid_y"})

            gdf_list.append(gdf)
        except Exception as e:
            logger.error(f"❌ 加载文件失败: {path} | 错误: {str(e)}")
            continue

    # 空数据检查
    if not gdf_list:
        logger.critical("❌ 无有效输入数据，终止处理")
        return

    # 加载配置参数
    config = load_weights()
    weights_dict = config.get('lcsm_weights', {
        "transition_frequency": 0.5,
        "contact_density": 0.3,
        "mixture_index": 0.2
    })
    min_dist, max_dist = config.get('lcsm_distance_range', [10, 300])
    decay_dict = config.get('distance_decay', {'a': 1000, 'b': 0.5})
    decay_a = decay_dict.get('a', 1000)
    decay_b = decay_dict.get('b', 0.5)
    impact_decay_factor = config.get('impact_decay_factor', 0.01)

    # 超时安全计算指标
    try:
        logger.info("⏳ 开始计算转化频率指标 (T)...")
        check_memory(1.5)
        df_t = run_with_timeout(compute_transition_frequency, (gdf_list, resolution), 7200)

        logger.info("⏳ 开始计算边界密度指标 (B)...")
        check_memory(1.5)
        df_b = run_with_timeout(compute_boundary_density, (gdf_list,), 7200)

        logger.info("⏳ 开始计算混合指数指标 (M)...")
        check_memory(1.5)
        df_m = run_with_timeout(compute_mixing_index, (gdf_list, resolution), 7200)
    except TimeoutError as e:
        logger.critical(f"❌ {str(e)}，终止计算")
        return
    except Exception as e:
        logger.critical(f"❌ 指标计算失败: {str(e)}")
        return

    # 合并指标（使用outer join并填充0）
    logger.info("🔗 合并指标表...")
    df = pd.merge(df_t, df_b, on=['class_a', 'class_b'], how='outer', suffixes=('', '_b'))
    df = pd.merge(df, df_m, on=['class_a', 'class_b'], how='outer', suffixes=('', '_m'))

    # 重命名列（避免后缀冲突）
    col_mapping = {'T': 'T', 'B': 'B', 'M': 'M'}
    df = df.rename(columns=col_mapping)

    # 填充NaN值
    df.fillna({'T': 0, 'B': 0, 'M': 0}, inplace=True)

    # 添加同类地物对
    logger.info("➕ 添加同类地物对...")
    all_codes = set()
    for df_part in [df_t, df_b, df_m]:
        all_codes.update(df_part['class_a'].unique())
        all_codes.update(df_part['class_b'].unique())

    same_class_pairs = []
    for code in all_codes:
        if not ((df['class_a'] == code) & (df['class_b'] == code)).any():
            same_class_pairs.append({
                'class_a': code, 'class_b': code,
                'T': 1.0, 'B': 1.0, 'M': 1.0
            })

    if same_class_pairs:
        df = pd.concat([df, pd.DataFrame(same_class_pairs)], ignore_index=True)

    # 归一化处理
    logger.info("📊 执行指标归一化...")
    for col in ['T', 'B', 'M']:
        min_val = df[col].min()
        max_val = df[col].max()
        range_val = max_val - min_val
        if range_val < 1e-6:
            logger.warning(f"⚠️ 指标 {col} 缺乏变化，归一化设置为0.5")
            df[f"{col}_norm"] = 0.5
        else:
            df[f"{col}_norm"] = (df[col] - min_val) / range_val

    # 加权合成耦合强度
    logger.info("⚖️ 计算加权耦合强度 (LCS_norm)...")
    df["LCS_norm"] = (
            weights_dict["transition_frequency"] * df["T_norm"] +
            weights_dict["contact_density"] * df["B_norm"] +
            weights_dict["mixture_index"] * df["M_norm"]
    )

    # 核心转换逻辑
    logger.info("⚙️ 计算物理距离 (LCS_distance)...")
    df["LCS_distance"] = max_dist - df["LCS_norm"] * (max_dist - min_dist)

    logger.info("🧪 转换作用强度 (impact_strength)...")
    df["impact_strength"] = decay_a * np.power(df["LCS_distance"] + 1e-6, -decay_b)
    df["impact_strength_exp"] = np.exp(-impact_decay_factor * df["LCS_distance"])

    # 同类特殊处理
    same_class_mask = (df['class_a'] == df['class_b'])
    df.loc[same_class_mask, 'LCS_distance'] = 0
    df.loc[same_class_mask, 'impact_strength'] = 1.0
    df.loc[same_class_mask, 'impact_strength_exp'] = 1.0

    # 保存结果
    pair_path = output_dir / "lcsm_pairs.csv"
    matrix_path = output_dir / "lcsm_matrix.csv"
    df.to_csv(pair_path, index=False)
    logger.info(f"💾 已保存LCSM关系对表: {pair_path}")

    save_lcsm_matrix(df, "impact_strength", matrix_path)
    distance_matrix_path = output_dir / "lcsm_distance_matrix.csv"
    save_lcsm_matrix(df, "LCS_distance", distance_matrix_path)
    logger.info(f"📏 已保存距离矩阵至 {distance_matrix_path}")
    log_memory_usage()