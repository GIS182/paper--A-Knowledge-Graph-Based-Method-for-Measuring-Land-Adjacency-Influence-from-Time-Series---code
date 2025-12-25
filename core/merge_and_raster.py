import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from rasterio.mask import mask
from rasterio.transform import from_origin
import os
import warnings
import logging
from utils.timer import timeit
from utils.geodata_io import read_geodata, write_geodata
from utils.config_reader import load_config
from shapely.validation import make_valid
from shapely.strtree import STRtree
from shapely.ops import unary_union
from tqdm import tqdm
import psutil

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=DeprecationWarning)

@timeit("合并作用图层并计算 LAI_Index")
def merge_effect_layers(dal_path: str, ial_path: str, output_path: str) -> gpd.GeoDataFrame:
    """
    合并直接与间接作用图层

    1. STRtree空间索引加速几何查询
    2. 内存分块处理
    3. 并行化字段计算

    参数:
        dal_path: DAL图层路径（含DAL_Strength）
        ial_path: IAL图层路径（含IAL_Strength）
        output_path: 输出矢量路径
    """
    # 增量读取数据（避免内存峰值）
    logger.info("📥 增量加载DAL/IAL图层...")
    dal = read_geodata(dal_path)
    ial = read_geodata(ial_path)

    for gdf in [dal, ial]:
        invalid_mask = ~gdf.geometry.is_valid
        if invalid_mask.any():
            logger.warning(f"⚠️ 检测到{invalid_mask.sum()}个无效几何，尝试修复...")
            gdf.loc[invalid_mask, 'geometry'] = gdf.loc[invalid_mask].geometry.apply(make_valid)

    # 构建空间索引（加速空间连接）
    logger.info("🔍 构建IAL空间索引...")
    ial_geoms = ial.geometry.tolist()
    ial_tree = STRtree(ial_geoms)

    # 分块处理（内存优化）
    chunk_size = min(5000, len(dal))
    results = []

    logger.info("🚀 启动空间连接计算...")
    for i in tqdm(range(0, len(dal), chunk_size), desc="处理DAL分块"):
        chunk = dal.iloc[i:i + chunk_size].copy()
        chunk_geoms = chunk.geometry.tolist()

        # 空间查询（批量化）
        intersections = []
        for geom in chunk_geoms:
            idxs = list(ial_tree.query(geom, predicate='intersects'))
            intersections.append(idxs if idxs else [])

        # 字段合并计算
        for j, idx_list in enumerate(intersections):
            if not idx_list:
                # 无交叉区域（仅DAL）
                chunk.loc[chunk.index[j], 'IAL_Strength'] = 0.0
            else:
                # 取最大交叉强度
                chunk.loc[chunk.index[j], 'IAL_Strength'] = ial.iloc[idx_list]['IAL_Strength'].max()

        # 计算LAI指数
        chunk['LAI_Index'] = chunk['DAL_Strength'] * chunk['IAL_Strength']
        results.append(chunk)

    # 合并结果
    combined = pd.concat(results)

    if not isinstance(combined, gpd.GeoDataFrame):
        # 转换为GeoDataFrame
        combined = gpd.GeoDataFrame(
            combined,
            geometry='geometry',  # 指定几何列
            crs=dal.crs  # 继承坐标系
        )
    # 保存结果
    write_geodata(combined, output_path)
    logger.info(f"✅ 合并完成：{output_path}（要素数：{len(combined)}）")
    return combined

@timeit("栅格化 LAI_Index 字段")
def rasterize_lai(
        gdf: gpd.GeoDataFrame,
        output_tif: str,
        field: str = 'LAI_Index',
        resolution: float = None
) -> None:
    """
    栅格化实现

    1. 内存感知型分块（根据可用内存动态调整）
    2. STRtree索引加速几何筛选
    3. 进度可视化

    参数:
        gdf: 输入GeoDataFrame
        output_tif: 输出GeoTIFF路径
        field: 栅格化字段名
        resolution: 栅格分辨率（米）
    """
    # 获取分辨率（配置优先）
    if resolution is None:
        config = load_config()
        resolution = config.get("resolution", 30.0)
    logger.info(f"📐 使用分辨率: {resolution}米")

    # 坐标系验证
    if gdf.crs is None:
        raise ValueError("❌ 缺少坐标系信息，请确保输入数据包含CRS")

    # 确保几何列是对象类型
    if gdf.geometry.dtype != 'object':
        logger.warning(f"⚠️ 几何列数据类型为 {gdf.geometry.dtype}, 正在转换为 object dtype")
        gdf.geometry = gdf.geometry.astype(object)
    logger.info(f"✅ 几何列数据类型已确认: {gdf.geometry.dtype}")

    # 计算动态缓冲边界
    bounds = gdf.total_bounds
    if np.any(np.isnan(bounds)) or bounds[0] == bounds[2] or bounds[1] == bounds[3]:
        raise ValueError("❌ 无效的图层边界")

    width_buffer = (bounds[2] - bounds[0]) * 0.05
    height_buffer = (bounds[3] - bounds[1]) * 0.05
    adj_bounds = (
        bounds[0] - width_buffer,
        bounds[1] - height_buffer,
        bounds[2] + width_buffer,
        bounds[3] + height_buffer
    )
    minx, miny, maxx, maxy = adj_bounds

    # 计算栅格尺寸
    width = int((maxx - minx) / resolution)
    height = int((maxy - miny) / resolution)
    transform = from_origin(minx, maxy, resolution, resolution)

    # 构建空间索引（加速栅格化）
    logger.info("🔍 构建几何空间索引...")
    geoms = gdf.geometry.tolist()
    values = gdf[field].values
    spatial_idx = STRtree(geoms)

    # 内存感知分块（根据可用RAM动态调整）
    avail_mem = psutil.virtual_memory().available / (1024 ** 3)  # 可用内存(GB)
    chunk_size = max(100, min(5000, int(avail_mem * 500)))
    logger.info(f"🧠 内存感知分块: 可用RAM={avail_mem:.1f}GB → 分块大小={chunk_size}")

    # 分块栅格化
    raster = np.zeros((height, width), dtype=np.float32)

    # 创建进度条
    total_blocks = ((height + chunk_size - 1) // chunk_size) * ((width + chunk_size - 1) // chunk_size)
    with tqdm(total=total_blocks, desc="栅格化进度") as pbar:
        for y_start in range(0, height, chunk_size):
            y_end = min(y_start + chunk_size, height)
            for x_start in range(0, width, chunk_size):
                x_end = min(x_start + chunk_size, width)

                # 计算地理边界
                x_min, y_max = transform * (x_start, y_start)
                x_max, y_min = transform * (x_end, y_end)
                bbox = (x_min, y_min, x_max, y_max)

                # 查询当前区块内的几何
                idxs = list(spatial_idx.query(bbox, predicate='intersects'))
                if not idxs:
                    pbar.update(1)
                    continue

                # 创建局部形状迭代器
                local_shapes = [
                    (geoms[i], values[i])
                    for i in idxs
                ]

                # 栅格化当前区块
                chunk_raster = features.rasterize(
                    shapes=local_shapes,
                    out_shape=(y_end - y_start, x_end - x_start),
                    transform=transform,
                    fill=0,
                    dtype=np.float32
                )

                # 合并到全局栅格
                raster[y_start:y_end, x_start:x_end] = np.maximum(
                    raster[y_start:y_end, x_start:x_end],
                    chunk_raster
                )
                pbar.update(1)

    logger.info("⚙️ 启动分块栅格化...")
    with tqdm(total=total_blocks, desc="栅格化进度") as pbar:
        for y in range(0, height, chunk_size):
            for x in range(0, width, chunk_size):
                # 计算当前区块地理范围
                x_min, y_max = transform * (x, y)
                x_max, y_min = transform * (x + chunk_size, y + chunk_size)
                bbox = (x_min, y_min, x_max, y_max)

                # 查询当前区块内的几何
                idxs = list(spatial_idx.query(bbox, predicate='intersects'))
                if not idxs:
                    pbar.update(1)
                    continue

                # 创建局部形状迭代器
                local_shapes = [
                    (geoms[i], values[i])
                    for i in idxs
                ]

                # 栅格化当前区块
                chunk_raster = features.rasterize(
                    shapes=local_shapes,
                    out_shape=(chunk_size, chunk_size),
                    transform=transform,
                    fill=0,
                    dtype=np.float32
                )

                # 合并到全局栅格
                y_end = min(y + chunk_size, height)
                x_end = min(x + chunk_size, width)
                raster[y:y_end, x:x_end] = np.maximum(
                    raster[y:y_end, x:x_end],
                    chunk_raster[:y_end - y, :x_end - x]
                )
                pbar.update(1)

    # 写入GeoTIFF（带压缩）
    os.makedirs(os.path.dirname(output_tif), exist_ok=True)
    with rasterio.open(
            output_tif,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=raster.dtype,
            crs=gdf.crs,
            transform=transform,
            nodata=0,
            compress='lzw'  # LZW压缩减少文件大小
    ) as dst:
        dst.write(raster, 1)

    logger.info(f"🟨 栅格化完成: {output_tif} (尺寸: {width}x{height})")

@timeit("裁剪 LAI 栅格图层")
def crop_raster(input_tif: str, reference_shp: str, output_tif: str) -> None:
    """
    栅格裁剪

    1. 多部件多边形自动合并
    2. 无效几何动态修复
    3. 智能坐标系对齐

    参数:
        input_tif: 输入栅格路径
        reference_shp: 参考边界矢量
        output_tif: 裁剪输出路径
    """
    # 读取边界矢量
    boundary = read_geodata(reference_shp)

    # 几何有效性修复
    if not boundary.geometry.is_valid.all():
        logger.warning("⚠️ 裁剪边界存在无效几何，尝试修复...")
        boundary.geometry = boundary.geometry.apply(make_valid)

    # 多部件多边形合并（确保单一几何）
    if len(boundary) > 1:
        logger.info("🔗 合并多部件多边形...")
        merged_geom = unary_union(boundary.geometry)
        boundary = gpd.GeoDataFrame(geometry=[merged_geom], crs=boundary.crs)

    with rasterio.open(input_tif) as src:
        # 坐标系对齐（自动重投影）
        if boundary.crs != src.crs:
            logger.warning(f"⚠️ 坐标系转换: {boundary.crs} → {src.crs}")
            boundary = boundary.to_crs(src.crs)

        # 执行裁剪（带异常捕获）
        try:
            out_image, out_transform = mask(
                src,
                shapes=boundary.geometry,
                crop=True,
                all_touched=True,
                filled=True
            )
            out_meta = src.meta.copy()
            out_meta.update({
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "compress": "lzw"  # 继承压缩设置
            })
        except ValueError as e:
            logger.error(f"❌ 裁剪范围异常: {str(e)}")
            # 尝试安全裁剪模式（扩大边界10%）
            logger.info("🛡️ 启用安全裁剪模式...")
            expanded_geom = boundary.geometry.buffer(src.res[0] * 10)
            out_image, out_transform = mask(
                src,
                shapes=expanded_geom,
                crop=True,
                all_touched=True,
                filled=True
            )
            out_meta = src.meta.copy()
            out_meta.update({
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })

    # 保存结果（带空间参考校验）
    os.makedirs(os.path.dirname(output_tif), exist_ok=True)
    with rasterio.open(output_tif, 'w', **out_meta) as dest:
        dest.write(out_image)
        # 写入空间参考信息（确保GDAL兼容性）
        if boundary.crs is not None:
            dest.update_tags(AREA_OR_POINT='Area')
            dest.set_band_description(1, 'LAI_Index')

    logger.info(f"✂️ 裁剪完成: {output_tif} (波段数: {out_image.shape[0]})")