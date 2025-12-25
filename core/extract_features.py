import geopandas as gpd
import numpy as np
import logging
from pathlib import Path
from typing import Union
from utils.timer import timeit
from utils.geodata_io import read_geodata, write_geodata

logger = logging.getLogger(__name__)

@timeit("几何特征计算")
def compute_geometry_features(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    为每个图斑高效计算核心几何特征：
    - area_ha: 面积（公顷）
    - perimeter_km: 边界长度（公里）
    - centroid_x/y: 重心坐标（原始CRS单位）

    1. 向量化操作替代循环
    2. 避免重复计算几何属性
    3. 添加几何有效性校验
    """

    invalid_mask = ~gdf.geometry.is_valid
    if invalid_mask.any():
        logger.warning(f"⚠️ 发现{invalid_mask.sum()}个无效几何，尝试自动修复...")
        gdf.loc[invalid_mask, 'geometry'] = gdf.loc[invalid_mask].geometry.buffer(0)

    # 向量化计算（单位转换）
    # 面积：m² → 公顷 (1ha=10,000m²)
    gdf['area_ha'] = gdf.geometry.area / 10000.0

    # 避免重复计算：一次性获取所有长度
    # 边界长度：m → 公里
    gdf['perimeter_km'] = gdf.geometry.length / 1000.0

    # 重心坐标（避免重复计算centroid）
    centroids = gdf.geometry.centroid
    gdf['centroid_x'] = centroids.x
    gdf['centroid_y'] = centroids.y

    logger.info(f"📐 几何特征计算完成 → 面积范围: {gdf['area_ha'].min():.2f}~{gdf['area_ha'].max():.2f} 公顷")
    return gdf

@timeit("特征提取流水线")
def extract_features(input_path: Union[str, Path], output_path: Union[str, Path]) -> gpd.GeoDataFrame:
    """
    特征提取主流程：
    1. 读取数据 → 2. 计算特征 → 3. 保存结果

    1. 使用Path对象处理路径
    2. 内存敏感型操作
    3. 异常安全封装
    """
    # 路径标准化
    input_path = Path(input_path)
    output_path = Path(output_path).with_suffix('.gpkg')
    output_path.parent.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在

    try:

        gdf = read_geodata(str(input_path))
        logger.info(f"📥 加载图斑数据: {input_path.name} → {len(gdf)}个要素")

        # 执行特征计算
        gdf = compute_geometry_features(gdf)

        layer_name = output_path.stem  # 使用文件名作为图层名
        write_geodata(gdf, str(output_path), layer=layer_name)

        logger.info(f"💾 保存特征数据: {output_path.name}")
        return gdf

    except Exception as e:
        logger.critical(f"❌ 特征提取失败: {str(e)}", exc_info=True)
        raise RuntimeError(f"特征提取失败: {input_path}") from e