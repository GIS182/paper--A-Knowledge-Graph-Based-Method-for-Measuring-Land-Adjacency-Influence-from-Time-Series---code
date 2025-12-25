import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from typing import Optional, Union, Dict, Callable
import logging

# 初始化日志
logger = logging.getLogger(__name__)

# 设置全局SCI论文字体格式
plt.rcParams.update({
    'font.family': 'serif',        # 使用衬线字体
    'font.serif': 'Times New Roman',  # 指定Times New Roman
})

# 热力图通用配置类
class MatrixHeatmapConfig:
    """
    1. 值转换函数（支持LAIM距离→强度转换）
    2. 大型矩阵自适应降采样
    3. 多格式输出支持
    """

    def __init__(
            self,
            title: str = "地类关系热力图",
            cbar_label: str = "耦合强度",
            cmap: str = "coolwarm",
            figsize: tuple = (10, 8),
            annot_size: int = 12,
            label_size: int = 14,
            title_size: int = 18,
            dpi: int = 600,
            linewidth: float = 0.5,
            linecolor: str = "gray",
            output_formats: tuple = ("tif", "jpg"),  # 只输出TIFF和JPG
            alpha: float = 1.0,  # 透明度控制
            annot_format: str = ".2f"  # 注释格式参数
    ):
        self.title = title
        self.cbar_label = cbar_label
        self.cmap = sns.color_palette(cmap, as_cmap=True)
        self.figsize = figsize
        self.annot_size = annot_size
        self.label_size = label_size
        self.title_size = title_size
        self.dpi = dpi
        self.linewidth = linewidth
        self.linecolor = linecolor
        self.output_formats = output_formats
        self.alpha = alpha
        self.annot_format = annot_format  # 保存注释格式

# 配色方案：viridis、summer、cool、coolwarm、Spectral、vlag、turbo
# LAIM/LCSM专用配置预设
LAIM_CONFIG = MatrixHeatmapConfig(
    title="Land-use Adjacency Distance Map(LAIM)",
    cbar_label="LAI value",
    cmap="summer",
    output_formats=("tif", "jpg"),
    annot_format=".1f"  # LAIM保留一位小数
)

LCSM_CONFIG = MatrixHeatmapConfig(
    title="Land-use Coupling Strength Map(LCSM)",
    cbar_label="LCS value",
    cmap="Spectral",
    output_formats=("tif", "jpg"),
    annot_format=".3f"  # LCSM保留三位小数
)

def _optimize_for_large_matrix(df: pd.DataFrame, config: MatrixHeatmapConfig) -> tuple:
    """
    大型矩阵优化策略：
    1. 自动降采样（>100×100）
    2. 动态调整注释密度
    3. 智能色阶范围
    """
    n = df.shape[0]

    # 降采样逻辑
    if n > 100:
        sample_rate = max(0.2, 100 / n)
        logger.warning(f"⚠️ 大型矩阵({n}×{n})启用降采样: {sample_rate:.1%}")
        df = df.iloc[::int(1 / sample_rate), ::int(1 / sample_rate)]

    # 注释密度优化
    config.annot_size = max(6, 12 - int(n / 20))
    annot_flag = n <= 30  # 仅在小矩阵显示数值

    # 自动色阶范围检测
    vmin, vmax = df.min().min(), df.max().max()
    if vmax - vmin < 1e-5:
        logger.warning("⚠️ 矩阵值变化过小，自动扩展色阶")
        vmin, vmax = vmin - 0.1, vmax + 0.1

    return df, annot_flag, vmin, vmax


def visualize_matrix_heatmap(
        matrix_csv_path: Union[str, Path],
        output_dir: Union[str, Path],
        config: MatrixHeatmapConfig,
        custom_labels: Optional[Dict[str, str]] = None
) -> None:
    """
    矩阵热力图可视化

    1. 值转换预处理
    2. 大型矩阵优化
    3. 多格式输出支持
    """
    # 路径安全处理
    matrix_csv_path = Path(matrix_csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = matrix_csv_path.stem

    logger.info(f"📊 加载矩阵数据: {matrix_csv_path}")
    try:
        df = pd.read_csv(matrix_csv_path, index_col=0)
        logger.info(f"✅ 矩阵加载成功 ({df.shape[0]}×{df.shape[1]})")

        # 空矩阵检测
        if df.isnull().all().all() or df.empty:
            raise ValueError("❌ 空值矩阵，无有效数据")
    except Exception as e:
        logger.critical(f"❌ 矩阵加载失败: {str(e)}")
        raise RuntimeError(f"矩阵文件读取错误: {matrix_csv_path}") from e

    # 应用自定义标签
    if custom_labels:
        logger.info("🏷️ 应用自定义地类标签")
        df.index = df.index.map(lambda x: custom_labels.get(x, x))
        df.columns = df.columns.map(lambda x: custom_labels.get(x, x))

    # 大型矩阵优化
    df, annot_flag, vmin, vmax = _optimize_for_large_matrix(df, config)

    # 创建热力图
    plt.figure(figsize=config.figsize)

    # 判断是否为LCSM矩阵（通过cbar_label识别）
    is_lcsm = config.cbar_label == "LCS value"

    heatmap = sns.heatmap(
        df,
        annot=annot_flag,  # 动态注释控制
        fmt=config.annot_format,  # 使用配置中的注释格式
        cmap=config.cmap,
        square=True,
        cbar_kws={"label": config.cbar_label},
        linewidths=config.linewidth,
        linecolor=config.linecolor,
        annot_kws={"size": config.annot_size},
        alpha=config.alpha,  # 透明度
        center=1 if is_lcsm else None,
        vmin=vmin,  # 动态色阶
        vmax=vmax
    )

    # 设置标题和标签
    plt.title(config.title, fontsize=config.title_size, pad=10, fontfamily='serif')
    plt.xlabel("Land Use Type", fontsize=config.label_size, fontfamily='serif')
    plt.ylabel("Land Use Type", fontsize=config.label_size, fontfamily='serif')
    plt.xticks(rotation=45, ha='right', fontsize=config.label_size - 2, fontfamily='serif')
    plt.yticks(rotation=0, fontsize=config.label_size - 2, fontfamily='serif')
    plt.tight_layout()

    # 确保颜色条标签也使用Times New Roman
    cbar = heatmap.collections[0].colorbar
    if cbar:
        cbar.ax.tick_params(labelsize=11)
        for label in cbar.ax.get_yticklabels():
            label.set_family('Times New Roman')

        # 设置颜色条标题的字体大小和字体
        cbar.set_label(config.cbar_label, fontsize=12, fontfamily='Times New Roman')
    plt.tight_layout()

    # 多格式输出
    for fmt in config.output_formats:
        output_path = output_dir / f"{base_name}_heatmap.{fmt}"
        plt.savefig(
            output_path,
            dpi=config.dpi,
            bbox_inches='tight',
            transparent=False  # TIFF和JPG格式不需要透明背景
        )
        logger.info(f"💾 {fmt.upper()}格式热力图已保存: {output_path}")

    plt.close()

# 专用函数接口（保持兼容性）
def visualize_laim_matrix(
        matrix_csv_path: Union[str, Path],
        output_dir: Union[str, Path],
        custom_labels: Optional[Dict[str, str]] = None
) -> None:
    """LAIM专用热力图接口"""
    visualize_matrix_heatmap(
        matrix_csv_path,
        output_dir,
        config=LAIM_CONFIG,
        custom_labels=custom_labels
    )

def visualize_lcsm_matrix(
        matrix_csv_path: Union[str, Path],
        output_dir: Union[str, Path],
        custom_labels: Optional[Dict[str, str]] = None
) -> None:
    """LCSM专用热力图接口"""
    visualize_matrix_heatmap(
        matrix_csv_path,
        output_dir,
        config=LCSM_CONFIG,
        custom_labels=custom_labels
    )

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # 创建输出目录
    os.makedirs("output/viz", exist_ok=True)

    # 地类标签映射
    land_class_labels = {
        "1": "Cropland",
        "2": "Forest",
        "3": "Shrub",
        "4": "Grassland",
        "5": "Water",
        "7": "Barren",
        "8": "Impervious"
    }

    # 生成LAIM热力图
    visualize_laim_matrix(
        matrix_csv_path="\data\laim_matrix.csv",
        output_dir="output/viz/laim",  # 输出目录
        custom_labels=land_class_labels  # 应用地类标签
    )

    # 生成LCSM热力图
    visualize_lcsm_matrix(
        matrix_csv_path="\data\lcsm_matrix.csv",
        output_dir="output/viz/lcsm",  # 输出目录
        custom_labels=land_class_labels
    )