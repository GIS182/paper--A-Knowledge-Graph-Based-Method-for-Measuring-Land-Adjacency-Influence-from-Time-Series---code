import os
import sys
from pathlib import Path
import logging
import json
import time
import traceback
import pandas as pd
import geopandas as gpd
import numpy as np
from tqdm import tqdm
import psutil
import gc
import multiprocessing as mp
import threading
from datetime import datetime
from utils.timer import timeit
from utils.config_reader import load_config
from core.preprocess import preprocess_shapefile
from core.classify_landuse import classify_by_landuse, load_class_map
from core.extract_features import extract_features
from core.build_laim import build_laim
from core.build_lcsm import build_lcsm
from core.adjacency_detector import detect_adjacency_pairs
from core.compute_effect import (
    generate_direct_effect_layer,
    generate_indirect_effect_layer,
    assign_effect_strength
)
from core.merge_and_raster import rasterize_lai, crop_raster, merge_effect_layers

logger = logging.getLogger(__name__)

def log_memory_usage():
    """记录当前内存使用情况"""
    mem = psutil.virtual_memory()
    logger.info(
        f"💾 内存使用: {mem.used / 1024 ** 3:.2f}GB/{mem.total / 1024 ** 3:.2f}GB (可用: {mem.available / 1024 ** 3:.2f}GB)")

# 读取参数
config = load_config()
INPUT_ROOT = Path(config["input_root"])
OUTPUT_ROOT = Path(config["output_root"])
YEARS = config["years"]
LAIM_YEARS = config["laim_years"]
LCSM_YEARS = config["lcsm_years"]
TARGET_YEAR = str(config["target_year"])
RESOLUTION = config["resolution"]

# 预创建所有输出目录
output_dirs = ["standardized", "classified", "features",
               "laim", "lcsm", "adjacency", "effect", "final", "logs"]
for d in output_dirs:
    os.makedirs(f"{OUTPUT_ROOT}/{d}", exist_ok=True)

# 加载地类编码映射
class_map = load_class_map()

# 存储分类图层和 features 路径（供后续 LAIM 和 LCSM 构建）
classified_dict = {}
features_dict = {}

@timeit("数据预处理（所有年份）")
def preprocess_all_years():
    """S1：标准化处理流水线（所有年份）"""
    logger.info("🔄 开始所有年份的标准化处理...")
    # 从配置读取图层前缀
    LAYER_PREFIX = config.get("gpkg_layer_prefix", "xfdnlanduse_")
    total_time = 0

    for year in tqdm(YEARS, desc="📅 处理年份"):
        y = str(year)
        start_time = time.time()

        # 构建完整图层路径
        layer_name = f"{LAYER_PREFIX}{y}"
        input_path = f"{INPUT_ROOT}|layername={layer_name}"

        # 标准化输出路径
        std_dir = OUTPUT_ROOT / "standardized" / y
        std_dir.mkdir(exist_ok=True)
        std_path = std_dir / f"landuse_{y}.gpkg"

        try:
            # 预处理（标准化+修复）
            preprocess_shapefile(input_path, str(std_path))
            logger.info(f"✅ {y}年标准化成功")
        except Exception as e:
            logger.error(f"❌ {y}年标准化失败: {str(e)}")
            continue

        year_time = time.time() - start_time
        total_time += year_time
        logger.info(f"⏱️ {y}年预处理用时: {round(year_time, 2)}秒")

    logger.info(f"✅ 所有年份预处理完成! 总用时: {round(total_time, 2)}秒")
    return total_time

@timeit("特征提取（所有年份）")
def extract_features_all_years():
    """S2：分类+特征提取流水线（所有年份）"""
    logger.info("🔄 开始所有年份的分类与特征提取...")
    total_time = 0

    for year in tqdm(YEARS, desc="📅 处理年份"):
        y = str(year)
        start_time = time.time()

        # 标准化数据路径
        std_path = OUTPUT_ROOT / "standardized" / y / f"landuse_{y}.gpkg"
        # 分类输出路径
        class_dir = OUTPUT_ROOT / "classified" / y
        # 特征输出路径
        feat_dir = OUTPUT_ROOT / "features" / y
        feat_dir.mkdir(exist_ok=True)
        feat_path = feat_dir / f"features_{y}.gpkg"

        try:
            # 分类（使用显式图层名）
            classify_by_landuse(str(std_path), str(class_dir), class_map)
            # 特征提取
            extract_features(str(std_path), str(feat_path))
            logger.info(f"✅ {y}年分类与特征提取成功")
            # 存储路径引用
            classified_dict[y] = class_dir
            features_dict[y] = feat_path
        except Exception as e:
            logger.error(f"❌ {y}年分类与特征提取失败: {str(e)}")
            continue

        year_time = time.time() - start_time
        total_time += year_time
        logger.info(f"⏱️ {y}年特征提取用时: {round(year_time, 2)}秒")

    logger.info(f"✅ 所有年份特征提取完成! 总用时: {round(total_time, 2)}秒")
    return total_time

def main():
    all_times = []
    stage_logs = []
    start_time = time.time()

    timeout_sec = 18000
    timeout_event = threading.Event()

    def timeout_handler():
        """超时处理函数"""
        timeout_event.set()
        raise TimeoutError("全局计算超时")

    # 创建定时器但不立即启动
    timer = threading.Timer(timeout_sec, timeout_handler)

    # 预加载地类映射表（确保只加载一次）
    class_map = load_class_map()

    try:
        timer.start()  # 启动超时计时器

        # ======= 环节1: 数据预处理（所有年份） =======
        stage_start = time.time()
        logger.info("🧹 开始数据预处理（所有年份）...")
        preprocess_time = preprocess_all_years()
        stage_time = round(time.time() - stage_start, 2)
        stage_logs.append({"stage": "data_preprocessing", "time_sec": stage_time})
        logger.info(f"✅ 数据预处理完成! 总用时: {stage_time}秒")

        # ======= 环节2: 特征提取（所有年份） =======
        stage_start = time.time()
        logger.info("🔍 开始地类分类与特征提取（所有年份）...")
        feature_time = extract_features_all_years()
        stage_time = round(time.time() - stage_start, 2)
        stage_logs.append({"stage": "feature_extraction", "time_sec": stage_time})
        logger.info(f"✅ 特征提取完成! 总用时: {stage_time}秒")

        # ======= 环节3: 知识图谱构建 =======
        stage_start = time.time()
        logger.info("🧠 构建知识图谱...")

        # 构建LAIM图谱
        logger.info("📊 构建 LAIM 图谱（邻接影响）...")
        laim_classified_paths = [str(classified_dict[str(y)]) for y in LAIM_YEARS]
        build_laim(laim_classified_paths, str(OUTPUT_ROOT / "laim"))

        # 构建LCSM图谱
        logger.info("📊 构建 LCSM 图谱（耦合强度）...")
        lcsm_feature_paths = [str(features_dict[str(y)]) for y in LCSM_YEARS]
        build_lcsm(lcsm_feature_paths, str(OUTPUT_ROOT / "lcsm"))

        stage_time = round(time.time() - stage_start, 2)
        stage_logs.append({"stage": "knowledge_graph", "time_sec": stage_time})
        logger.info(f"✅ 知识图谱构建完成! 用时: {stage_time}秒")

        # ======= 环节4: 邻接关系识别 =======
        stage_start = time.time()
        logger.info("🔍 识别邻接关系...")
        # 动态路径构建
        feat_path = features_dict[TARGET_YEAR]
        adj_dir = OUTPUT_ROOT / "adjacency"
        adj_dir.mkdir(exist_ok=True)
        adj_path = adj_dir / f"{TARGET_YEAR}_adjacency.csv"

        # 执行邻接关系检测
        detect_adjacency_pairs(str(feat_path), str(adj_path))

        stage_time = round(time.time() - stage_start, 2)
        stage_logs.append({"stage": "adjacency_detection", "time_sec": stage_time})
        logger.info(f"✅ 邻接关系识别完成! 用时: {stage_time}秒")

        # ======= 环节5: 邻接作用计算 =======
        stage_start = time.time()
        logger.info("🧮 计算邻接作用...")
        # 准备其他路径
        std_path = OUTPUT_ROOT / "standardized" / TARGET_YEAR / f"landuse_{TARGET_YEAR}.gpkg"
        effect_dir = OUTPUT_ROOT / "effect"
        effect_dir.mkdir(exist_ok=True)
        dal_raw = effect_dir / f"{TARGET_YEAR}_dal_raw.gpkg"
        dal_final = effect_dir / f"{TARGET_YEAR}_dal.gpkg"
        ial_raw = effect_dir / f"{TARGET_YEAR}_ial_raw.gpkg"
        ial_final = effect_dir / f"{TARGET_YEAR}_ial.gpkg"
        final_dir = OUTPUT_ROOT / "final"
        final_dir.mkdir(exist_ok=True)
        merged_path = final_dir / f"{TARGET_YEAR}_linjie.gpkg"
        laim_path = OUTPUT_ROOT / "laim" / "laim_pairs.csv"
        lcsm_path = OUTPUT_ROOT / "lcsm" / "lcsm_pairs.csv"

        # 执行计算链
        generate_direct_effect_layer(str(feat_path), str(adj_path), str(laim_path), str(dal_raw))
        assign_effect_strength(str(dal_raw), str(lcsm_path), str(dal_final), "DAL_Strength")
        generate_indirect_effect_layer(str(feat_path), str(laim_path), str(lcsm_path), str(ial_raw))
        assign_effect_strength(str(ial_raw), str(lcsm_path), str(ial_final), "IAL_Strength")
        merge_effect_layers(str(dal_final), str(ial_final), str(merged_path))

        stage_time = round(time.time() - stage_start, 2)
        stage_logs.append({"stage": "effect_computation", "time_sec": stage_time})
        logger.info(f"✅ 邻接作用计算完成! 用时: {stage_time}秒")

        # ======= 环节6: 结果栅格化 =======
        stage_start = time.time()
        logger.info("🖨️ 结果栅格化...")
        raster_path = final_dir / f"{TARGET_YEAR}_lai_raw.tif"
        raster_crop_path = final_dir / f"{TARGET_YEAR}_lai_cropped.tif"

        merged_gdf = gpd.read_file(str(merged_path))
        rasterize_lai(merged_gdf, str(raster_path), resolution=RESOLUTION)
        crop_raster(str(raster_path), str(std_path), str(raster_crop_path))

        stage_time = round(time.time() - stage_start, 2)
        stage_logs.append({"stage": "rasterization", "time_sec": stage_time})
        logger.info(f"✅ 结果栅格化完成! 用时: {stage_time}秒")

        # 总计时
        elapsed = round(time.time() - start_time, 2)
        all_times.append({
            "year": "all",
            "status": "Success",
            "runtime_sec": elapsed,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        logger.info(f"🏁 全流程完成! 总用时: {elapsed}秒")

    except TimeoutError as e:
        # 捕获超时异常
        error_msg = f"全局计算超时: {str(e)}"
        logger.error(error_msg)
        all_times.append({
            "year": "all",
            "status": "Failed",
            "error": str(e),
            "runtime_sec": round(time.time() - start_time, 2),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        log_path = OUTPUT_ROOT / "logs" / "failure.log"
        with open(str(log_path), "w", encoding='utf-8') as f:
            f.write(error_msg)
        logger.error(f"[ERROR] 流程失败: {error_msg}")
    except Exception as e:
        # 错误处理
        error_msg = traceback.format_exc()
        all_times.append({
            "year": "all",
            "status": "Failed",
            "error": str(e),
            "runtime_sec": round(time.time() - start_time, 2),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        log_path = OUTPUT_ROOT / "logs" / "failure.log"
        with open(str(log_path), "w", encoding='utf-8') as f:
            f.write(error_msg)
        clean_error = str(e).replace('❌', '[ERROR]')
        logger.error(f"[ERROR] 流程失败: {clean_error}")
    finally:
        # 取消超时计时器
        timer.cancel()

    # 保存日志（Path对象兼容）
    runtime_log = OUTPUT_ROOT / "logs" / "runtime_summary.csv"
    stage_log = OUTPUT_ROOT / "logs" / "stage_times.csv"
    pd.DataFrame(all_times).to_csv(str(runtime_log), index=False)
    pd.DataFrame(stage_logs).to_csv(str(stage_log), index=False)
    logger.info(f"📊 已保存运行时间日志: {runtime_log} 和 {stage_log}")
    log_memory_usage()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()