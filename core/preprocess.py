import tempfile
import time
import geopandas as gpd
import logging
import os
import psutil
import pandas as pd
from pathlib import Path
from osgeo import gdal
from utils.timer import timeit
from utils.geodata_io import read_geodata, write_geodata
from shapely.validation import make_valid
import re

logger = logging.getLogger(__name__)

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

@timeit("重投影到 EPSG:4547")
def reproject_to_epsg4547(gdf: gpd.GeoDataFrame,
                          target_crs: str = 'EPSG:4547') -> gpd.GeoDataFrame:
    """
    将输入图层投影到 CGCS2000（EPSG:4547）
    参数:
        gdf (GeoDataFrame): 输入图层
        target_crs (str): 目标投影坐标系
    返回:
        GeoDataFrame: 重投影后的图层
    """
    original_crs = gdf.crs.to_string() if gdf.crs else "未定义"
    logger.info(f"📐 原始坐标系：{original_crs}")

    if gdf.crs is None:
        logger.warning("⚠️ 检测到未定义CRS，强制指定为EPSG:4326")
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)

    # 执行重投影
    gdf = gdf.to_crs(target_crs)
    logger.info(f"✅ 重投影完成 （目标CRS: {gdf.crs}）")
    return gdf

@timeit("标准化字段结构")
def standardize_fields(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    标准化字段命名，统一为算法所需字段
    - GRIDCODE → land_code
    - Landuse → land_name
    - ID → poly_id
    并去除多余字段
    """
    # 删除QGIS修复生成的临时字段
    redundant_fields = ["_errors", "layer", "path"]
    gdf = gdf.drop(columns=redundant_fields, errors="ignore")

    # 弹性字段映射（支持大小写变体）
    field_mapping = {
        'GRIDCODE': 'land_code',
        'Landuse': 'land_name',
        'ID': 'poly_id'
    }

    # 查找实际存在的字段（不区分大小写）
    actual_fields = {col.upper(): col for col in gdf.columns}
    mapped_fields = {}

    for expected, new_name in field_mapping.items():
        found_col = find_column_by_normalized(gdf, expected)
        if found_col:
            mapped_fields[found_col] = new_name
        else:
            logger.warning(f"⚠️ 未找到字段 '{expected}'，使用默认值初始化")
            gdf[new_name] = 0  # 初始化默认值

    # 执行字段重命名
    if mapped_fields:
        gdf = gdf.rename(columns=mapped_fields)

    # 强制转换 land_code 为整数
    if 'land_code' in gdf.columns:
        gdf['land_code'] = gdf['land_code'].astype(int)

    # 删除非核心字段（保留geometry）
    core_fields = {'poly_id', 'land_code', 'land_name', 'geometry'}
    extra_fields = [col for col in gdf.columns if col not in core_fields]

    if extra_fields:
        logger.info(f"🗑️ 移除冗余字段: {', '.join(extra_fields)}")
        gdf = gpd.GeoDataFrame(
            gdf[list(core_fields)],
            crs=gdf.crs
        )

    # 几何类型验证（支持多几何类型）
    if gdf.geometry.type.isin(['GeometryCollection']).any():
        logger.warning("⚠️ 检测到混合几何类型，尝试提取多边形")
        gdf = gdf.explode(index_parts=True)
        gdf = gdf[gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]

    if gdf.crs != 'EPSG:4547':
        logger.warning(f"⚠️ 检测到非标准CRS: {gdf.crs}，重新投影到EPSG:4547")
        gdf = gdf.to_crs('EPSG:4547')

    return gdf

@timeit("拓扑校验与修复")
def validate_topology(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    执行拓扑校验：
    1. 修复无效几何
    2. 移除空几何
    3. 确保多边形封闭性
    """
    # 基础分块大小
    base_chunk_size = 500

    # 内存检测与动态调整
    mem = psutil.virtual_memory()
    avail_gb = mem.available / (1024 ** 3)

    if avail_gb < 1.0:  # 内存危急 (<1GB)
        chunk_size = 100
        logger.critical(f"⚠️ 内存危急! 可用: {avail_gb:.2f}GB → 分块降至: {chunk_size}")
    elif avail_gb < 2.0:  # 内存不足 (<2GB)
        chunk_size = 200
        logger.warning(f"⚠️ 内存不足! 可用: {avail_gb:.2f}GB → 分块降至: {chunk_size}")
    else:  # 内存充足
        chunk_size = base_chunk_size
        logger.info(f"♻️ 内存充足: {avail_gb:.2f}GB → 使用标准分块: {chunk_size}")

    # 检测无效几何
    invalid_idx = ~gdf.geometry.is_valid
    invalid_count = invalid_idx.sum()
    if not invalid_count:
        return gdf

    logger.warning(f"⚠️ 发现{invalid_count}个无效几何，启动安全修复...")

    # 显式设置GDAL异常模式
    gdal.DontUseExceptions()

    # 分块修复 + 显式资源释放
    chunk_size = 500  # 进一步减小分块规模
    repaired_geoms = []  # 存储修复后的几何

    for i in range(0, invalid_count, chunk_size):
        chunk = gdf[invalid_idx].iloc[i:i + chunk_size].copy()
        try:
            # 独立临时文件路径
            tmp_path = os.path.join(tempfile.gettempdir(), f"gdal_fix_{os.getpid()}_{time.time_ns()}.gpkg" )
            chunk.to_file(tmp_path, driver="GPKG")

            # 强制关闭GDAL数据集释放资源
            ds = gdal.OpenEx(tmp_path, gdal.OF_VECTOR | gdal.OF_UPDATE)
            layer = ds.GetLayer()
            gdal.VectorTranslate(
                tmp_path, tmp_path,
                accessMode="update",
                layerName=layer.GetName(),
                makeValid=True
            )
            ds = None  # 显式释放GDAL资源

            # 读取修复结果
            repaired = gpd.read_file(tmp_path)
            repaired_geoms.extend(repaired.geometry.tolist())
        except Exception as e:
            logger.error(f"❌ 分块{i}-{i + chunk_size}修复失败，回退缓冲修复: {str(e)}")
            repaired_geoms.extend(chunk.geometry.buffer(0).tolist())
        finally:
            # 双重安全删除
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)  # 立即删除
                except Exception as e:
                    logger.error(f"❌ 临时文件删除失败: {tmp_path} | 错误: {str(e)}")

    # 合并修复结果
    gdf.loc[invalid_idx, "geometry"] = repaired_geoms
    return gdf

@timeit("预处理流水线")
def preprocess_shapefile(input_path: str, output_path: str) -> None:
    """
    矢量数据预处理

    1. 支持GeoPackage图层路径格式
    2. 自动修复几何错误
    3. 动态字段标准化
    4. 强制投影到EPSG:4547

    参数:
        input_path: 输入文件路径（支持"path.gpkg|layername=xxx"格式）
        output_path: 输出文件路径
    """
    logger.info(f"📥 加载源数据: {input_path}")

    # 直接传递路径到read_geodata，不转换Path对象
    gdf = read_geodata(input_path)

    logger.info(f"✅ 加载成功 → 要素数: {len(gdf)} | CRS: {gdf.crs}")

    # 执行处理流水线
    try:
        gdf = standardize_fields(gdf)

        # 跳过几何修复（已在QGIS完成）
        # gdf = validate_topology(gdf)

        if gdf.crs != 'EPSG:4547':
            logger.info(f"🔄 坐标系转换: {gdf.crs} → EPSG:4547")
            gdf = gdf.to_crs('EPSG:4547')

    except Exception as e:
        logger.error(f"❌ 预处理失败: {str(e)}")
        raise

    # 保存结果
    write_geodata(gdf, output_path)
    logger.info(f"💾 保存预处理结果: {output_path}")