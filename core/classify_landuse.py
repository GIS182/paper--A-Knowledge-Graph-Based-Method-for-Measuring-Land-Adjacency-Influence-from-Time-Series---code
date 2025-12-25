import geopandas as gpd
import logging
from pathlib import Path
import json
import os
from typing import Dict, Union
from utils.timer import timeit
from utils.geodata_io import read_geodata, write_geodata

logger = logging.getLogger(__name__)

_class_map_cache = None

@timeit("加载地类映射表")
def load_class_map(json_path: Union[str, Path] = "config/class_map.json") -> Dict[str, str]:
    """
    加载地类编码到名称的映射
    参数:
        json_path: JSON配置文件路径
    返回:
        地类映射字典 {代码: 名称}
    """
    global _class_map_cache

    # 如果已加载则直接返回缓存
    if _class_map_cache is not None:
        #  仅主进程记录日志
        if os.getpid() == os.getppid():
            logger.info(f"♻️ 使用缓存地类映射表 ({len(_class_map_cache)}个地类)")
        return _class_map_cache

    json_path = Path(json_path)
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        _class_map_cache = data  # 设置缓存

        # 仅主进程记录日志
        if os.getpid() == os.getppid():
            logger.info(f"✅ 加载地类映射表: {json_path.name} → {len(data)}个地类")
        return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.critical(f"❌ 地类映射表加载失败: {json_path} | 错误: {str(e)}")
        raise RuntimeError(f"地类映射表加载失败: {str(e)}") from e

@timeit("地类分类输出")
def classify_by_landuse(
        input_path: Union[str, Path],
        output_dir: Union[str, Path],
        class_map: Dict[str, str]
) -> None:
    """
    按照 land_code 对图斑进行分类，输出多个地类图层

    参数:
        input_path: 标准化图层路径
        output_dir: 输出目录 (如 output/classified/2013/)
        class_map: 地类编码 → 地类名称映射
    """
    # 统一路径处理
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    # 创建输出目录（确保只创建一次）
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 创建输出目录: {output_dir}")

    try:
        # 统一使用geodata_io接口（显式指定图层名为文件名）
        gdf = read_geodata(str(input_path), layer=input_path.stem)
        logger.info(f"📥 加载标准化图层: {input_path.name} → {len(gdf)}个图斑")

        # 关键字段校验
        for field in ['land_code', 'geometry']:
            if field not in gdf.columns:
                raise ValueError(f"❌ 缺失关键字段 '{field}'，请检查预处理结果")

        try:
            gdf['land_code'] = gdf['land_code'].astype(int)
        except (ValueError, TypeError):

            logger.warning("⚠️ 检测到非整型地类编码，尝试映射转换...")
            # 构建编码映射字典（将字符串映射到整数）
            code_mapping = {str(k): int(k) for k in class_map.keys()}
            # 对于非数字的编码，我们保留原字符串，但会记录警告
            unknown_codes = set()

            def map_code(code):
                if isinstance(code, str) and code in code_mapping:
                    return code_mapping[code]
                else:
                    try:
                        return int(code)
                    except:
                        unknown_codes.add(str(code))
                        return -9999  # 无效编码占位符

            gdf['land_code'] = gdf['land_code'].apply(map_code)
            if unknown_codes:
                logger.warning(f"⚠️ 发现无法映射的地类编码: {unknown_codes}，已标记为-9999")
            # 检查是否有无效编码
            if (gdf['land_code'] == -9999).any():
                invalid_count = (gdf['land_code'] == -9999).sum()
                logger.error(f"❌ 存在 {invalid_count} 个无效地类编码，无法分类")

        # 分类处理
        valid_classes = 0
        for code_str, name in class_map.items():
            try:
                # 统一转换为整数进行比较
                code = int(code_str)
                class_gdf = gdf[gdf['land_code'] == code]

                if not class_gdf.empty:
                    # 安全文件名处理（替换特殊字符）
                    safe_name = name.replace(' ', '_').replace('/', '_')

                    layer_name = f"class_{code}_{safe_name}"
                    output_path = output_dir / f"{layer_name}.gpkg"

                    write_geodata(class_gdf, str(output_path), layer=layer_name)

                    valid_classes += 1
                    logger.info(f"✅ 输出地类图层: {output_path.name} ({len(class_gdf)}个图斑)")
                else:
                    logger.warning(f"⚠️ 地类 {code}-{name} 无图斑，跳过")
            except ValueError:
                logger.error(f"❌ 无效地类编码: {code_str}，跳过该分类")

        logger.info(f"🏁 分类完成: 共输出 {valid_classes}/{len(class_map)} 个有效地类图层")

    except Exception as e:
        logger.critical(f"❌ 地类分类失败: {input_path} | 错误: {str(e)}")
        raise RuntimeError(f"地类分类失败: {str(e)}") from e