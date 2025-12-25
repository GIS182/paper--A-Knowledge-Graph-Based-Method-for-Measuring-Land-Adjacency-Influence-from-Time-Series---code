import geopandas as gpd
import fiona
import os
import logging
from pathlib import Path
from typing import Union
import numpy as np
import gc
import psutil
import time
import re

logger = logging.getLogger(__name__)

def get_gpkg_layers(gpkg_path: str) -> list:
    """获取GPKG文件中的所有图层名称"""
    try:
        return fiona.listlayers(gpkg_path)
    except Exception as e:
        logger.error(f"❌ 读取图层列表失败: {gpkg_path} | 错误: {str(e)}")
        return []

# 内存监控函数
def check_memory(min_avail_gb=1.0):
    mem = psutil.virtual_memory()
    avail_gb = mem.available / 1024**3
    if avail_gb < min_avail_gb:
        logger.warning(f"🛑 可用内存低于{min_avail_gb}GB ({avail_gb:.2f}GB)，触发GC回收")
        gc.collect()
    return mem.available

def read_geodata(path: Union[str, Path], layer: str = None) -> gpd.GeoDataFrame:
    """
    地理数据读取器

    1. 支持 "path.gpkg|layername=xxx" 格式的输入路径
    2. 自动处理图层名称解析
    3. 智能CRS检测与修复

    参数:
        path: 文件路径（支持特殊格式）
        layer: 备用图层名（优先级低于路径中的图层名）
    返回:
        GeoDataFrame（自动修复CRS）
    """
    check_memory(1.0)

    path_str = str(path)
    logger.info(f"📥 加载地理数据: {path_str}")

    # 解析特殊格式路径
    if "|layername=" in path_str:
        gpkg_path, layer_name = path_str.split("|layername=")
        logger.info(f"🔍 解析GeoPackage图层: {gpkg_path} → {layer_name}")
        gdf = gpd.read_file(gpkg_path, layer=layer_name)

        # 自动修复缺失CRS
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4547", allow_override=True)
            logger.warning(f"⚠️ 缺失CRS，已强制设置为EPSG:4547")
        return gdf

    # 标准路径处理
    path_obj = Path(path_str)
    if not path_obj.exists():
        raise FileNotFoundError(f"路径不存在: {path_str}")

    try:
        # 支持GeoJSON格式
        if path_obj.suffix.lower() in ['.geojson', '.json']:
            logger.info(f"🌐 加载GeoJSON数据: {path_obj.name}")
            gdf = gpd.read_file(path_str)
            # 自动精度优化
            for col in gdf.select_dtypes(include='float64').columns:
                gdf[col] = gdf[col].astype(np.float32)
            return gdf

        if path_obj.suffix.lower() == '.gpkg':
            if layer:
                logger.info(f"🔍 使用指定图层: {layer}")
                gdf = gpd.read_file(path_str, layer=layer)
            else:
                available_layers = fiona.listlayers(str(path_obj))
                target_layer = layer or path_obj.stem

                matched_layers = [lyr for lyr in available_layers if lyr == target_layer]
                if not matched_layers:
                    # 尝试模糊匹配（忽略大小写和特殊字符）
                    normalized_target = re.sub(r'[^a-z0-9]', '', target_layer.lower())
                    matched_layers = [lyr for lyr in available_layers
                                      if re.sub(r'[^a-z0-9]', '', lyr.lower()) == normalized_target]

                if matched_layers:
                    logger.info(f"🔍 找到匹配图层: {matched_layers[0]}")
                    gdf = gpd.read_file(path_str, layer=matched_layers[0])
                else:
                    raise ValueError(f"❌ 未找到匹配图层，可用图层: {', '.join(available_layers)}")

            for col in gdf.select_dtypes(include='float64').columns:
                gdf[col] = gdf[col].astype(np.float32)
            return gdf

        elif path_obj.suffix.lower() == '.shp':
            logger.info(f"📤 读取Shapefile: {path_obj}")
            try:
                with fiona.open(path_str) as src:
                    encoding = src.encoding or 'GBK'
                return gpd.read_file(path_str, encoding=encoding)
            except UnicodeDecodeError:
                logger.warning("⚠️ GBK编码失败，尝试UTF-8")
                return gpd.read_file(path_str, encoding='UTF-8')

        else:
            raise ValueError(f"❌ 不支持的格式: {path_obj.suffix}")

    except Exception as e:
        layer_info = f"图层: {layer} | " if layer else ""
        error_msg = f"读取失败 [{path_str}] {layer_info}| 错误: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e

def write_geodata(gdf: gpd.GeoDataFrame, path: Union[str, Path], layer: str = None):
    """
    地理数据写入器

    1. 自动创建父目录
    2. 智能数据类型转换
    3. 统一UTF-8编码

    参数:
        gdf: 待写入的GeoDataFrame
        path: 输出路径
        layer: GPKG专用图层名（可选）
    """
    # 内存警戒检查
    check_memory(1.0)

    path_obj = Path(path) if isinstance(path, str) else path
    logger.info(f"💾 保存地理数据: {path_obj} | 图层: {layer if layer else '默认'}")
    os.makedirs(path_obj.parent, exist_ok=True)

    try:
        # 通用字段类型转换
        for col in gdf.columns:
            # 字符串类型统一处理
            if gdf[col].dtype == object:
                gdf[col] = gdf[col].astype(str)
            # 浮点类型精度优化
            elif 'float' in str(gdf[col].dtype):
                gdf[col] = gdf[col].astype(np.float32)

        # 支持GeoJSON格式
        if path_obj.suffix.lower() in ['.geojson', '.json']:
            gdf.to_file(path_obj, driver='GeoJSON', encoding='UTF-8')
            return

        if path_obj.suffix.lower() == '.gpkg':
            if not layer:
                layer = path_obj.stem
                logger.warning(f"⚠️ 未指定图层名，使用默认值: {layer}")

            # GPKG写入优化
            gdf.to_file(
                path_obj,
                driver='GPKG',
                layer=layer,
                encoding='UTF-8',
                index=False
            )

        elif path_obj.suffix.lower() == '.shp':
            # Shapefile字段名截断处理
            gdf.columns = [col[:10] for col in gdf.columns]
            gdf.to_file(path_obj, encoding='GBK')

        else:
            raise ValueError(f"❌ 不支持的格式: {path_obj.suffix}")

    except Exception as e:
        layer_info = f"图层: {layer} | " if layer else ""
        error_msg = f"写入失败 [{path_obj}] {layer_info}| 错误: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e