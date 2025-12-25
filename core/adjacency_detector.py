import pandas as pd
import numpy as np
import os
import warnings
import logging
from tqdm import tqdm
from utils.timer import timeit
from utils.geodata_io import read_geodata
from shapely.geometry import LineString, MultiLineString
from multiprocessing import Pool, cpu_count, current_process

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=DeprecationWarning)

BUFFER_DISTANCE = 150

def _init_worker():
    logging.getLogger().handlers = []
    warnings.simplefilter("ignore")
    os.environ["OMP_NUM_THREADS"] = "1"
    current_process().name = f"AdjWorker-{current_process().pid}"

def _process_chunk(args):
    """
    多进程处理函数：计算单个图斑的邻接关系
    """
    idx, geom, poly_id, land_code, candidate_idxs, gdf_geoms, gdf_ids, gdf_codes = args
    records = []

    if not hasattr(geom, 'is_valid') or geom.is_empty or not geom.is_valid:
        return records

    try:
        buffer_geom = geom.buffer(BUFFER_DISTANCE)  # 仅当几何有效时创建缓冲区
        if buffer_geom.is_empty or not buffer_geom.is_valid:
            return records
    except Exception:
        return records

    for candidate_idx in candidate_idxs:
        if candidate_idx <= idx:  # 避免重复计算
            continue

        neighbor_geom = gdf_geoms[candidate_idx]
        neighbor_id = gdf_ids[candidate_idx]
        neighbor_code = gdf_codes[candidate_idx]

        # 验证邻接几何有效性
        if not hasattr(neighbor_geom, 'is_valid') or neighbor_geom.is_empty or not neighbor_geom.is_valid:
            continue

        # 缓冲区预判：快速排除非邻接图斑
        if not buffer_geom.intersects(neighbor_geom):
            continue

        # 计算邻接关系
        if geom.touches(neighbor_geom):
            try:
                shared_geom = geom.intersection(neighbor_geom)

                # 几何类型和有效性校验
                if (
                        shared_geom.is_empty
                        or not hasattr(shared_geom, 'length')
                        or not isinstance(shared_geom, (LineString, MultiLineString))
                ):
                    continue

                shared_length = shared_geom.length
                if shared_length > 0:
                    # 根据ID顺序同步调整地类编码
                    if poly_id < neighbor_id:
                        records.append((poly_id, neighbor_id, land_code, neighbor_code, shared_length))
                    else:
                        records.append((neighbor_id, poly_id, neighbor_code, land_code, shared_length))
            except Exception as e:
                continue  # 单点失败不影响整体

    return records

@timeit("识别图斑邻接对（共享边界）")
def detect_adjacency_pairs(input_path: str, output_path: str, num_processes: int = None) -> None:
    """
    识别邻接图斑对

    1. 使用数组存储几何和属性，加速读取
    2. 分块多进程并行计算
    3. 基于R-tree的精确空间查询

    参数:
        input_path: 输入features图层路径
        output_path: 输出CSV路径
        num_processes: 进程数（默认使用全部核心-1）
    """
    # 读取数据并预处理
    gdf = read_geodata(input_path)
    if gdf.empty:
        logger.warning("⚠️ 输入图层为空，跳过邻接检测")
        return

    # 构建全局索引数据结构
    gdf = gdf.reset_index(drop=True)
    sindex = gdf.sindex  # R-tree空间索引
    gdf_geoms = gdf.geometry.values
    gdf_ids = gdf['poly_id'].values
    gdf_codes = gdf['land_code'].values

    # 动态调整进程数（预留内存）
    num_processes = num_processes or max(1, cpu_count() - 2)  # 减少进程数避免OOM
    chunks = []

    logger.info(f"🔍 生成空间查询任务（图斑数: {len(gdf)} | 进程数: {num_processes}）")
    for idx, (geom, poly_id, land_code) in enumerate(zip(gdf_geoms, gdf_ids, gdf_codes)):
        # 跳过无效几何
        if geom.is_empty or not geom.is_valid:
            continue

        # 获取候选邻接图斑索引
        candidate_idxs = list(sindex.intersection(geom.bounds))
        chunks.append((idx, geom, poly_id, land_code, candidate_idxs, gdf_geoms, gdf_ids, gdf_codes))

    # 多进程计算
    records = []
    logger.info(f"🚀 启动{num_processes}进程计算邻接关系")
    with Pool(
            processes=num_processes,
            initializer=_init_worker,  # 子进程环境隔离
            maxtasksperchild=50  # 降低子进程回收阈值（减少内存累积）
    ) as pool:
        # 动态分块大小（每进程约处理1万图斑）
        chunk_size = max(500, len(chunks) // (num_processes * 20))
        results = pool.imap_unordered(_process_chunk, chunks, chunksize=chunk_size)

        # 进度监控（实时更新）
        with tqdm(total=len(chunks), desc="邻接边界计算") as pbar:
            for result in results:
                if result:  # 跳过空结果
                    records.extend(result)
                pbar.update(1)

    # 构建结果DataFrame
    if records:
        df = pd.DataFrame(records, columns=[
            'poly_id_a', 'poly_id_b',
            'land_code_a', 'land_code_b',
            'shared_length'
        ])
        logger.info(f"📊 邻接关系统计: 共发现 {len(df)} 个有效邻接对")
    else:
        df = pd.DataFrame(columns=[
            'poly_id_a', 'poly_id_b',
            'land_code_a', 'land_code_b',
            'shared_length'
        ])
        logger.warning("⚠️ 未发现有效邻接关系")

    # 保存结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"💾 邻接对表已保存至: {output_path}")

def detect_adjacency_gpd(input_path, output_path):

    gdf = read_geodata(input_path)
    if gdf.empty:
        logger.warning("⚠️ 输入图层为空，跳过邻接检测")
        return

    # 生成邻接矩阵
    adj_matrix = gdf.geometry.apply(
        lambda g: gdf.geometry.touches(g)
    )

    # 提取邻接对
    rows, cols = np.where(np.triu(adj_matrix, k=1))  # 仅取上三角避免重复

    records = []
    for i, j in zip(rows, cols):
        geom_i = gdf.iloc[i].geometry
        geom_j = gdf.iloc[j].geometry
        shared_geom = geom_i.intersection(geom_j)

        # 增强几何校验
        if shared_geom.is_empty or not isinstance(shared_geom, (LineString, MultiLineString)):
            continue

        # 获取属性并排序
        id_i, id_j = gdf.iloc[i]['poly_id'], gdf.iloc[j]['poly_id']
        code_i, code_j = gdf.iloc[i]['land_code'], gdf.iloc[j]['land_code']

        if id_i < id_j:
            records.append({
                'poly_id_a': id_i, 'poly_id_b': id_j,
                'land_code_a': code_i, 'land_code_b': code_j,
                'shared_length': shared_geom.length
            })
        else:
            records.append({
                'poly_id_a': id_j, 'poly_id_b': id_i,
                'land_code_a': code_j, 'land_code_b': code_i,
                'shared_length': shared_geom.length
            })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"✅ 邻接对表已保存：{output_path}")
