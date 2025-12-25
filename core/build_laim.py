import geopandas as gpd
import pandas as pd
import numpy as np
import logging
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry import Point
from itertools import product
from utils.timer import timeit
from utils.geodata_io import read_geodata
import multiprocessing as mp
from tqdm import tqdm
import psutil
import time
import re
import gc
from shapely.strtree import STRtree
from concurrent.futures import ProcessPoolExecutor, as_completed
from shapely import distance

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
    加载LAIM权重配置
    """
    json_path = Path(json_path)
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            weights = json.load(f)['laim_weights']
        logger.info(f"✅ 加载LAIM权重: {weights}")
        return weights
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        logger.critical(f"❌ LAIM权重配置加载失败: {json_path} | 错误: {str(e)}")
        # 返回默认权重防止中断
        return {'avg_shortest_distance': 0.4, 'centroid_distance': 0.3, 'boundary_distance': 0.3}

def validate_input(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """动态修复缺失字段（兼容旧数据）并确保坐标系一致"""
    # 1. 查找land_code字段（兼容不同命名）
    land_code_col = find_column_by_normalized(gdf, "land_code")
    if not land_code_col:
        raise KeyError("❌ 数据中未找到land_code字段")

    # 重命名统一字段名
    if land_code_col != "land_code":
        gdf = gdf.rename(columns={land_code_col: "land_code"})

    # 2. 强制统一数据类型
    gdf['land_code'] = gdf['land_code'].astype(int)

    # 3. 坐标系统一为EPSG:4547
    if gdf.crs != 'EPSG:4547':
        logger.warning(f"⚠️ 重投影至EPSG:4547 (原始CRS: {gdf.crs})")
        gdf = gdf.to_crs('EPSG:4547')

    # 4. 自动计算缺失的重心坐标
    if 'centroid_x' not in gdf.columns or 'centroid_y' not in gdf.columns:
        logger.warning("⚠️ 自动修复：缺少重心坐标字段，重新计算...")
        gdf['centroid_x'] = gdf.geometry.centroid.x
        gdf['centroid_y'] = gdf.geometry.centroid.y

    return gpd.GeoDataFrame(gdf, geometry='geometry')

# 计算两组重心之间的平均欧氏距离
def calculate_centroid_distance_matrix(centroids: Dict[int, np.ndarray]) -> Dict[Tuple[int, int], float]:
    """
    重心距离计算（避免O(n²)内存爆炸）
    使用KD树+抽样统计替代全矩阵计算
    """
    if not centroids:
        logger.error("❌ 空重心数据，无法计算距离")
        return {}

    logger.info(f"📏 计算重心距离矩阵 | 地类数: {len(centroids)}")
    centroid_distances = {}
    SAMPLE_SIZE = 500  # 每类最大采样点数量（控制内存）

    all_dists = []
    for code_i, points_i in centroids.items():
        if len(points_i) == 0:
            continue

        # 对大型地类抽样（避免内存爆炸）
        if len(points_i) > SAMPLE_SIZE:
            idx = np.random.choice(len(points_i), SAMPLE_SIZE, replace=False)
            points_i = points_i[idx]

        for code_j, points_j in centroids.items():
            if code_i == code_j or len(points_j) == 0:
                continue

            # 对大型地类抽样
            if len(points_j) > SAMPLE_SIZE:
                idx = np.random.choice(len(points_j), SAMPLE_SIZE, replace=False)
                points_j = points_j[idx]

            # 构建KD树查询最近邻
            tree = cKDTree(points_j.astype(np.float32))
            dists, _ = tree.query(points_i, k=1, workers=1)
            avg_dist = np.mean(dists)
            centroid_distances[(code_i, code_j)] = avg_dist
            all_dists.append(avg_dist)

    global_avg = np.mean(all_dists) if all_dists else 1000.0
    logger.info(f"✅ 生成 {len(centroid_distances)} 个距离关系 | 全局平均距离: {global_avg:.2f}m")
    return centroid_distances

# 重构边界距离计算函数
def extract_polygons(geoms: gpd.GeoSeries, sample_size: int = 100) -> List[Polygon]:
    """
    正确提取多边形边界（解决空边界问题）
    """
    valid_polys = []
    for geom in geoms:
        # 跳过无效几何
        if geom.is_empty or not geom.is_valid:
            continue

        # 处理MultiPolygon类型
        if geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                if isinstance(poly, Polygon) and not poly.is_empty:
                    valid_polys.append(poly)
        # 处理Polygon类型
        elif geom.geom_type == 'Polygon':
            valid_polys.append(geom)
        # 跳过其他类型
        else:
            logger.warning(f"⚠️ 跳过非多边形几何类型: {geom.geom_type}")
    return valid_polys[:min(len(valid_polys), sample_size)]

def _calc_pair_distance(args):
    """
    计算两个地类多边形之间的最短距离（替代共享边界）
    """
    code_i, code_j, tree_i, geoms_j = args
    try:
        if code_i == code_j:
            return (code_i, code_j), 0.0

        total_dist = 0.0
        count = 0
        for geom_j in geoms_j:
            # 严格几何验证
            if not isinstance(geom_j, (Polygon, MultiPolygon)):
                continue
            if geom_j.is_empty or not geom_j.is_valid:
                continue

            # 查询最近几何
            nearest_geom = tree_i.nearest(geom_j)

            # 验证最近几何有效性
            if not isinstance(nearest_geom, (Polygon, MultiPolygon)) or nearest_geom.is_empty:
                continue

            dist = distance(geom_j, nearest_geom)
            total_dist += dist
            count += 1

        return (code_i, code_j), total_dist / count if count > 0 else 0.0
    except Exception as e:
        logger.error(f"边界距离计算失败 {code_i}-{code_j}: {str(e)}")
        return (code_i, code_j), 0.0

# 计算边界间最近距离（所有边界对）
def calculate_boundary_distance_matrix(class_geoms: Dict[int, gpd.GeoSeries]) -> Dict[Tuple[int, int], float]:
    """
    使用空间索引优化边界距离计算
    策略：使用STRtree索引+抽样策略+多进程
    """
    if not class_geoms:
        logger.error("❌ 空几何数据，无法计算边界距离")
        return {}

    logger.info("📐 边界距离矩阵计算...")
    SAMPLE_SIZE = 100  # 每类最大几何样本量
    NUM_WORKERS = min(4, os.cpu_count())  # 控制进程数

    # 构建空间索引字典 {code: (STRtree, 几何列表)}
    spatial_data = {}
    for code, geoms in class_geoms.items():
        if not geoms.empty:
            polygons = extract_polygons(geoms, SAMPLE_SIZE)
            if polygons:
                spatial_data[code] = (STRtree(polygons), polygons)

    # 准备多进程任务
    tasks = []
    codes = list(spatial_data.keys())
    for code_i in codes:
        tree_i, geoms_i = spatial_data[code_i]
        for code_j in codes:
            _, geoms_j = spatial_data.get(code_j, (None, []))
            tasks.append((code_i, code_j, tree_i, geoms_j))

    # 多进程计算
    boundary_distances = {}
    logger.info(f"🚀 启动多进程计算 | 任务数: {len(tasks)} | 进程数: {NUM_WORKERS}")

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(_calc_pair_distance, task): task for task in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="边界距离计算"):
            (code_i, code_j), dist = future.result()
            boundary_distances[(code_i, code_j)] = dist

    logger.info(f"✅ 边界距离矩阵完成 | 组合数: {len(boundary_distances)}")
    return boundary_distances

# 计算平均最短邻接距离（重心之间的最近距离对）
def calculate_nearest_neighbor_matrix(class_centroids: Dict[int, np.ndarray], threshold=300) -> Dict[
    Tuple[int, int], float]:
    """
    批量计算所有地类对之间的平均最近邻距离
    避免重复构建KDTree
    nn_distances是一个字典，存储所有地类对之间的最近邻距离计算结果，键为 (code_i, code_j)，值为距离值。
    它是一个中间计算结果容器。nn_distances是字典（数据集合），而 avg_shortest_distance是浮点数（权重值）
    """
    if not class_centroids:
        logger.error("❌ 空重心数据，无法计算最近邻距离")
        return {}

    logger.info("🔍 最近邻距离矩阵计算...")
    nn_distances = {}

    # 1. 构建所有地类的KD树（避免重复构建）
    trees = {}
    for code, points in class_centroids.items():
        if points.size > 0:
            trees[code] = cKDTree(points.astype(np.float32))

    # 2. 计算全局平均距离（用于填充无效值）
    global_avg = 0.0
    valid_pairs = 0

    # 3. 计算地类对距离
    codes = list(trees.keys())
    for code_i in codes:
        tree_i = trees[code_i]
        for code_j in codes:
            if code_i == code_j:
                nn_distances[(code_i, code_j)] = 0.0
                continue

            tree_j = trees.get(code_j)
            if tree_j is None:
                continue

            # 查询code_i到code_j的最近邻
            dists, _ = tree_i.query(tree_j.data, k=1, distance_upper_bound=threshold)
            valid_dists = dists[dists < threshold]

            if valid_dists.size > 0:
                avg_dist = np.mean(valid_dists)
                nn_distances[(code_i, code_j)] = avg_dist
                global_avg += avg_dist
                valid_pairs += 1

    # 4. 处理无效值
    global_avg = global_avg / valid_pairs if valid_pairs > 0 else 150.0
    for pair in [(i, j) for i in codes for j in codes]:
        if pair not in nn_distances:
            nn_distances[pair] = global_avg

    return nn_distances

def save_laim_matrix(df: pd.DataFrame, value_col: str, output_path: Union[str, Path]) -> None:
    """
    保存为邻接矩阵格式（对角线置0）
    使用pivot_table替代循环
    """
    output_path = Path(output_path)
    logger.info("📐 正在转换为邻接影响矩阵格式...")

    # 关键字段存在性检查
    required_cols = ['class_a', 'class_b', value_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.critical(f"❌ 缺失关键字段: {missing_cols}，无法生成矩阵")
        return

    # 创建透视表
    pivot_df = df.pivot_table(index='class_a', columns='class_b', values=value_col, fill_value=0)

    # 确保所有地类都在行列中
    all_codes = sorted(set(df['class_a']).union(df['class_b']))
    pivot_df = pivot_df.reindex(index=all_codes, columns=all_codes, fill_value=0)

    # 对角线置零
    for code in all_codes:
        if code in pivot_df.index and code in pivot_df.columns:
            pivot_df.loc[code, code] = 0

    pivot_df.to_csv(output_path)
    logger.info(f"✅ 已保存地类邻接影响矩阵至 {output_path.name}")

@timeit("构建地类邻接影响图谱 LAIM")
def build_laim(input_dirs: List[Union[str, Path]], output_dir: Union[str, Path]) -> None:
    """
    构建 LAIM 图谱（支持多期数据）

        1. 增量加载数据，减少内存峰值
        2. 预聚合地类几何和重心
        3. 并行计算独立指标
    """
    log_memory_usage()
    weights = load_weights()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 步骤1: 增量加载数据并预聚合
    class_centroids = {}  # {land_code: numpy数组(N,2)}
    class_geoms = {}  # {land_code: GeoSeries}

    logger.info("📥 增量加载多期分类图层...")
    input_dirs = [Path(d) for d in input_dirs]

    # 获取所有有效土地类型（1-8，排除6）
    VALID_CODES = {1, 2, 3, 4, 5, 7, 8}

    def check_memory(min_avail_gb=1.0):
        """内存警戒线检查（非阻塞）"""
        mem = psutil.virtual_memory()
        avail_gb = mem.available / 1024**3
        if avail_gb < min_avail_gb:
            logger.warning(f"🛑 可用内存低于{min_avail_gb}GB ({avail_gb:.2f}GB)，触发GC回收")
            gc.collect()
            new_mem = psutil.virtual_memory()
            logger.info(f"♻️ GC后可用内存: {new_mem.available/1024**3:.2f}GB")
        return mem.available

    for dir_path in tqdm(input_dirs, desc="加载目录"):
        # 获取所有地类图层文件（支持.gpkg和.shp）
        files = list(dir_path.glob("*.gpkg")) + list(dir_path.glob("*.shp"))
        if not files:
            logger.warning(f"⚠️ 目录中未找到有效文件: {dir_path}")
            continue

        for file_path in files:
            try:
                # 移除阻塞性sleep，改为轻量级内存检查
                avail_mem = check_memory(1.0)
                if avail_mem < 1.5 * 1024 ** 3:
                    logger.warning(f"🛑 可用内存低于1.5GB ({avail_mem / 1024 ** 3:.2f}GB)，跳过当前文件")
                    continue  # 跳过当前文件但不阻塞

                layer_name = file_path.stem
                logger.info(f"🔍 加载文件: {file_path.name}|{layer_name}")
                gdf = read_geodata(str(file_path), layer=layer_name)
                gdf = validate_input(gdf)

                gdf['centroid_x'] = gdf['centroid_x'].astype(np.float32)
                gdf['centroid_y'] = gdf['centroid_y'].astype(np.float32)

                # 按地类分组聚合
                for land_code, group in gdf.groupby('land_code'):
                    if land_code not in VALID_CODES:
                        continue

                    # 获取当前地类的重心点集
                    centroids = group[['centroid_x', 'centroid_y']].values
                    if land_code not in class_centroids:
                        class_centroids[land_code] = centroids
                    else:
                        class_centroids[land_code] = np.vstack([class_centroids[land_code], centroids])

                    # 用concat替代append（解决GeoSeries无append方法）
                    if land_code not in class_geoms:
                        class_geoms[land_code] = group.geometry
                    else:
                        class_geoms[land_code] = pd.concat([class_geoms[land_code], group.geometry])

                # 及时释放内存
                del gdf
                gc.collect()

            except Exception as e:
                logger.error(f"❌ 加载文件失败: {file_path} | 错误: {str(e)}")
                continue

    # 过滤空数据
    valid_codes = [code for code in class_centroids if class_centroids[code].size > 0]
    class_centroids = {code: class_centroids[code] for code in valid_codes}
    class_geoms = {code: class_geoms[code] for code in valid_codes}
    logger.info(f"📊 有效地类数: {len(valid_codes)} | 地类列表: {valid_codes}")

    # 关键检查：有效地类数为0时终止
    if not valid_codes:
        logger.critical("❌ 致命错误：未发现有效地类，请检查输入数据")
        raise ValueError("输入数据中未找到有效的land_code字段或所有地类数据为空")

    # 步骤2: 计算三类空间指标
    logger.info("⏳ 计算地类间空间关系指标...")
    # 1. 先计算边界距离（需要几何数据）
    boundary_dists = calculate_boundary_distance_matrix(class_geoms)

    # 2. 再计算重心距离
    centroid_dists = calculate_centroid_distance_matrix(class_centroids)

    # 3. 最后计算最近邻距离（释放几何数据）
    nn_dists = calculate_nearest_neighbor_matrix(class_centroids)
    del class_centroids, class_geoms # 及时释放内存
    gc.collect()

    # 空结果检查
    if not centroid_dists or not boundary_dists or not nn_dists:
        logger.critical("❌ 空间指标计算失败，请检查数据有效性")
        raise RuntimeError("空间关系指标计算结果为空")

    # 步骤3: 合并结果
    results = []
    all_pairs = set(centroid_dists.keys()) | set(boundary_dists.keys()) | set(nn_dists.keys())
    for (class_a, class_b) in all_pairs:
        results.append({
            'class_a': class_a,
            'class_b': class_b,
            'centroid_distance': centroid_dists.get((class_a, class_b), 0),
            'boundary_distance': boundary_dists.get((class_a, class_b), 0),
            'avg_shortest_distance': nn_dists.get((class_a, class_b), 0)
        })

    df = pd.DataFrame(results)
    logger.info(f"📦 合并结果完成 | 关系对数: {len(df)}")

    # 空数据框检查
    if df.empty:
        logger.critical("❌ 合并结果为空，无法继续处理")
        raise ValueError("地类关系对数据框为空")

    # 步骤4: 归一化处理
    logger.info("📊 执行归一化处理...")

    # 字段存在性检查和自动补全
    required_cols = ['centroid_distance', 'boundary_distance', 'avg_shortest_distance']
    for col in required_cols:
        if col not in df.columns:
            logger.warning(f"⚠️ 缺失字段自动补全: {col}")
            df[col] = 0

    # 记录原始指标范围
    original_min_max = {}
    for col in required_cols:
        min_val = df[col].min()
        max_val = df[col].max()

        # 防止极端值影响归一化
        if max_val - min_val < 1e-6:
            min_val = 0
            max_val = max_val if max_val > 0 else 1.0

        original_min_max[col] = (min_val, max_val)
        df[f"{col}_norm"] = (df[col] - min_val) / (max_val - min_val + 1e-8)  # 避免除零

    # 步骤5: 加权合成LAI指数
    logger.info("⚖️ 执行权重加权叠加生成LAI_norm值...")
    df["LAI_norm"] = (
            weights["centroid_distance"] * df["centroid_distance_norm"] +
            weights["boundary_distance"] * df["boundary_distance_norm"] +
            weights["avg_shortest_distance"] * df["avg_shortest_distance_norm"]
    )

    # 步骤6: 反归一化
    logger.info("⚙️ 执行LAI值反归一化...")
    min_possible = sum(
        weights[col] * original_min_max[col][0]
        for col in required_cols
    )
    max_possible = sum(
        weights[col] * original_min_max[col][1]
        for col in required_cols
    )

    df["LAI"] = (df["LAI_norm"] * (max_possible - min_possible) + min_possible)

    # 步骤7: 保存结果
    pairs_path = output_dir / "laim_pairs.csv"
    matrix_path = output_dir / "laim_matrix.csv"
    lcs_matrix_path = output_dir / "lcsm_matrix.csv"

    df.to_csv(pairs_path, index=False)
    logger.info(f"✅ 已保存LAIM关系对表: {pairs_path.name}")

    save_laim_matrix(df, "LAI", matrix_path)
    log_memory_usage()