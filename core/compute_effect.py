import pandas as pd
import geopandas as gpd
import numpy as np
import os
import math
import json
import logging
from tqdm import tqdm
from rtree import index
from utils.geodata_io import read_geodata, write_geodata
from utils.timer import timeit
from shapely.geometry import Polygon
from shapely.strtree import STRtree
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import psutil
import time

logger = logging.getLogger(__name__)

@timeit("加载权重配置")
def load_decay_params(config_path="config/weight_config.json"):
    """加载距离衰减参数并添加异常处理"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            a = config['distance_decay']['a']
            b = config['distance_decay']['b']
            logger.info(f"✅ 加载距离衰减参数: a={a}, b={b} (公式: buffer_distance = a * exp(-b * strength))")
            return a, b
    except Exception as e:
        logger.critical(f"❌ 衰减参数加载失败: {str(e)}")
        return 1000, 0.5  # 默认值

def _process_batch(args):
    """独立函数确保可序列化（使用LAIM的LAI值）"""
    batch_records, features, adj_df, laim_dict, decay_a, decay_b = args
    results = []
    for _, row in batch_records.iterrows():
        a_id, b_id = row['poly_id_a'], row['poly_id_b']
        a_code, b_code = row['land_code_a'], row['land_code_b']

        try:
            # 获取几何
            geom_a = features.loc[a_id].geometry
            geom_b = features.loc[b_id].geometry

            # 使用LAIM的LAI值（邻接影响值）
            lai = laim_dict.get((a_code, b_code)) or laim_dict.get((b_code, a_code))
            if not lai:
                continue

            # 计算缓冲距离（使用LAI值）
            buffer_distance = decay_a * math.exp(-decay_b * lai)

            # 几何有效性检查
            if not geom_a.is_valid or not geom_b.is_valid:
                continue

            # 动态计算缓冲区分辨率
            area = max(geom_a.area, geom_b.area)
            resolution = 16 if area > 1e6 else 8

            # 添加缓冲区容差减少计算复杂度
            buffer_a = geom_a.buffer(buffer_distance, resolution=resolution, join_style=2)
            inter = buffer_a.intersection(geom_b)

            if inter.is_empty or inter.area < 1e-6:
                continue

            results.append({
                "source_id": a_id,
                "target_id": b_id,
                "source_code": a_code,
                "target_code": b_code,
                "geometry": inter,
                "buffer_distance": buffer_distance,
                "impact_strength": lai  # 使用LAI值
            })
        except Exception as e:
            logger.error(f"❌ 处理邻接对失败 {a_id}-{b_id}: {str(e)}")
    return results

@timeit("生成直接邻接作用图层 DAL")
def generate_direct_effect_layer(features_path, adjacency_csv, laim_csv, output_path):
    """
    生成直接邻接作用图层优化：
    1. 增强几何有效性检查
    2. 优化缓冲区分辨率设置
    3. 改进多进程任务调度
    """
    # 读取数据
    features = read_geodata(features_path).set_index("poly_id")
    adjacency_df = pd.read_csv(adjacency_csv)
    laim_df = pd.read_csv(laim_csv)  # 使用LAIM数据

    # 构建LAIM的LAI字典
    laim_dict = {}
    for _, row in laim_df.iterrows():
        key = (row["class_a"], row["class_b"])
        laim_dict[key] = row["LAI"]  # 使用LAI值

    # 获取衰减参数
    decay_a, decay_b = load_decay_params()

    # 内存监控函数
    def check_memory(min_avail_gb=1.0):
        mem = psutil.virtual_memory()
        avail_gb = mem.available / 1024 ** 3
        if avail_gb < min_avail_gb:
            logger.warning(f"🛑 可用内存低于{min_avail_gb}GB ({avail_gb:.2f}GB)，触发GC回收")
            gc.collect()
            time.sleep(1)
            new_mem = psutil.virtual_memory()
            logger.info(f"♻️ GC后可用内存: {new_mem.available / 1024 ** 3:.2f}GB")
        return mem.available

    # 动态调整分块大小
    total_records = len(adjacency_df)
    avail_mem = psutil.virtual_memory().available / (1024 ** 3)
    chunk_size = max(100, min(5000, total_records // max(1, int(avail_mem / 0.5))))

    logger.info(f"🧠 动态分块 | 记录数: {total_records} | 可用内存: {avail_mem:.1f}GB → 分块大小: {chunk_size}")

    # 多进程处理
    results = []
    cpu_count = os.cpu_count() or 4
    logger.info(f"🚀 启动{cpu_count}进程计算直接邻接作用...")

    with ProcessPoolExecutor(max_workers=cpu_count) as executor:
        futures = []
        for i in range(0, total_records, chunk_size):
            # 内存警戒检查
            check_memory(1.5)

            chunk = adjacency_df.iloc[i:i + chunk_size]
            futures.append(
                executor.submit(
                    _process_batch,
                    (chunk, features, adjacency_df, laim_dict, decay_a, decay_b)
                )
            )

        # 进度监控
        with tqdm(total=len(futures), desc="处理邻接对") as pbar:
            for future in as_completed(futures):
                try:
                    batch_results = future.result()
                    if batch_results:
                        results.extend(batch_results)
                except Exception as e:
                    logger.error(f"❌ 批处理失败: {str(e)}")
                finally:
                    pbar.update(1)
                    # 及时释放内存
                    del future
                    gc.collect()

    # 保存结果
    if results:
        gdf = gpd.GeoDataFrame(results, crs=features.crs)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        write_geodata(gdf, output_path)
        logger.info(f"✅ 成功生成 DAL 图层: {output_path} (要素数: {len(gdf)})")
        return output_path
    else:
        logger.warning("⚠️ 无有效相交图形，未生成 DAL 图层")
        return None

def _process_indirect_chunk(args):
    """独立函数处理间接邻接块"""
    chunk_indices, gdf, spatial_idx, lcsm_dict, decay_a, decay_b = args
    chunk_results = []
    for idx_a in chunk_indices:
        row_a = gdf.iloc[idx_a]
        geom_a = row_a.geometry
        a_code = row_a['land_code']

        # 空间查询候选集
        candidate_indices = spatial_idx.query(geom_a, predicate='intersects')
        for idx_b in candidate_indices:
            if idx_b == idx_a:
                continue

            row_b = gdf.iloc[idx_b]
            b_code = row_b['land_code']

            # 跳过同类地物
            if a_code == b_code:
                continue

            # 使用LCSM的作用强度
            strength = lcsm_dict.get((a_code, b_code), 0.01)

            # 距离衰减计算
            buffer_distance = decay_a * math.exp(-decay_b * strength)

            # 简化几何操作
            try:
                buffer_a = geom_a.buffer(
                    buffer_distance,
                    resolution=8,  # 固定分辨率减少计算量
                    join_style=2    # 斜接连接减少异常
                )
                inter = buffer_a.intersection(row_b.geometry)

                if not inter.is_empty and inter.area > (geom_a.area * 0.001):
                    chunk_results.append({
                        "source_id": row_a['poly_id'],
                        "target_id": row_b['poly_id'],
                        "source_code": a_code,
                        "target_code": b_code,
                        "geometry": inter,
                        "buffer_distance": buffer_distance,
                        "impact_strength": strength
                    })
            except Exception as e:
                logger.error(f"几何操作失败: {str(e)}")
    return chunk_results

@timeit("生成间接邻接作用图层 IAL")
def generate_indirect_effect_layer(features_path, laim_csv, lcsm_csv, output_path):
    """
    生成间接邻接作用图层优化：
    1. 使用LCSM的作用强度值（impact_strength）
    2. 优化空间索引查询效率
    3. 改进分块并行策略
    """
    # 1. 加载参数和数据集
    decay_a, decay_b = load_decay_params()
    gdf = read_geodata(features_path)
    total_features = len(gdf)
    logger.info(f"📥 加载特征数据: {features_path} → {total_features}个要素")

    # 2. 加载LCSM矩阵（使用impact_strength字段）
    lcsm_df = pd.read_csv(lcsm_csv)
    lcsm_dict = {}
    for _, row in lcsm_df.iterrows():
        key = (row['class_a'], row['class_b'])
        lcsm_dict[key] = row['impact_strength']

    # 3. 构建空间索引
    logger.info("🔍 构建空间索引...")
    spatial_idx = STRtree(gdf.geometry)

    # 内存监控函数
    def check_memory(min_avail_gb=1.0):
        mem = psutil.virtual_memory()
        if mem.available < min_avail_gb * 1024 ** 3:
            gc.collect()
            time.sleep(1)
            return psutil.virtual_memory().available
        return mem.available

    # 4. 动态分块并行计算
    logger.info("🚀 启动并行计算间接作用...")
    cpu_count = os.cpu_count() or 4

    # 基于内存的动态分块
    avail_mem = psutil.virtual_memory().available / (1024 ** 3)
    chunk_size = max(50, min(500, total_features // max(1, int(avail_mem / 0.5))))

    logger.info(f"🧠 动态分块 | 要素数: {total_features} | 可用内存: {avail_mem:.1f}GB → 分块大小: {chunk_size}")

    indices = list(range(total_features))
    results = []

    with ProcessPoolExecutor(max_workers=cpu_count) as executor:
        futures = []
        for i in range(0, total_features, chunk_size):
            # 内存警戒检查
            check_memory(1.5)

            chunk = indices[i:i + chunk_size]
            futures.append(executor.submit(
                _process_indirect_chunk,
                (chunk, gdf, spatial_idx, lcsm_dict, decay_a, decay_b)
            ))

        # 进度监控
        with tqdm(total=len(futures), desc="处理间接邻接") as pbar:
            for future in as_completed(futures):
                try:
                    chunk_results = future.result()
                    if chunk_results:
                        results.extend(chunk_results)
                except Exception as e:
                    logger.error(f"❌ 分块处理失败: {str(e)}")
                finally:
                    pbar.update(1)
                    # 及时释放内存
                    del future
                    gc.collect()

    # 5. 保存结果
    if results:
        gdf_out = gpd.GeoDataFrame(results, crs=gdf.crs)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        write_geodata(gdf_out, output_path)
        logger.info(f"✅ 生成IAL图层: {output_path} (要素数: {len(gdf_out)})")
        return output_path
    else:
        logger.warning("⚠️ 无有效间接作用区域")
        return None

@timeit("作用强度字段赋值")
def assign_effect_strength(effect_path, lcsm_csv, output_path, strength_field="impact_strength"):
    """
    1. 向量化操作替代apply
    2. 减少内存拷贝
    """
    gdf = read_geodata(effect_path)
    lcsm_df = pd.read_csv(lcsm_csv)

    # 向量化构建字典
    lcsm_dict = {}
    for _, row in lcsm_df.iterrows():
        key1 = (row['class_a'], row['class_b'])
        key2 = (row['class_b'], row['class_a'])
        lcsm_dict[key1] = row['impact_strength']
        lcsm_dict[key2] = row['impact_strength']

    # 向量化赋值替代apply
    # 创建临时列存储查询键
    gdf['strength_key'] = list(zip(gdf['source_code'], gdf['target_code']))

    # 使用map方法向量化赋值
    gdf[strength_field] = gdf['strength_key'].map(
        lambda x: lcsm_dict.get(x, 0.01)
    )

    # 删除临时列
    del gdf['strength_key']

    # 保存结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write_geodata(gdf, output_path)
    logger.info(f"✅ {strength_field} 赋值完成: {output_path}")
    return output_path
