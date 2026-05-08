import sys
import os
import matplotlib.cbook
import warnings

# 忽略警告
warnings.filterwarnings("ignore")

# 【核心修复】兼容性补丁
if not hasattr(matplotlib.cbook, "_Stack") and hasattr(matplotlib.cbook, "Stack"):
    matplotlib.cbook._Stack = matplotlib.cbook.Stack
# ================= 1. 导入库 =================
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ================= 1. 学术风格全局配置 =================
# 设置全局字体：中文使用黑体/宋体，英文使用 Times New Roman (学术必备)
plt.rcParams['font.sans-serif'] = ['SimSun', 'Times New Roman']
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['mathtext.fontset'] = 'stix'  # 使公式字体接近 Times New Roman

# 【核心修复】确保 SVG 中的文字被转换为路径，防止插入 Word 后字体缺失
plt.rcParams['svg.fonttype'] = 'path'
plt.rcParams['pdf.fonttype'] = 42

# 定义学术期刊常用的“色彩盲友好”配色方案
COLORS = {
    "Best": "#E31A1C",  # 红色 (本文创新点)
    "color0": "#1F78B4",  # 蓝色
    "color1": "#6A3D9A",  # 紫色
    "color2": "#33A02C",  # 绿色
    "Base": "#777777"  # 灰色 (基准线)
}


def plot_academic_reward(files_config, window_size=200):
    """
    生成符合论文出版标准的 Reward 对比图
    """
    # 按照 A4 纸半栏或全栏宽度设定尺寸 (单位：英寸)
    plt.figure(figsize=(9, 6), dpi=300)
    ax = plt.gca()

    for item in files_config:
        try:
            # 1. 读取并截取前 1000回合
            df = pd.read_excel(item["path"])
            df = df.iloc[:60000] if len(df) >= 60000 else df

            x = df['episode']
            y_raw = df['reward']

            # 2. 计算滑动平均和标准差 (用于绘制阴影)
            y_smooth = y_raw.rolling(window=window_size, min_periods=1).mean()
            y_std = y_raw.rolling(window=window_size, min_periods=1).std()

            # 3. 绘制阴影区域 (表示训练的方差/不确定性，比细线更专业)
            # 这种写法是顶会论文 (ICLR/NeurIPS) 的标准画法
            ax.fill_between(x, y_smooth - 0.5 * y_std, y_smooth + 0.5 * y_std,
                            color=item["color"], alpha=0.15, edgecolor='none')

            # 4. 绘制主平滑曲线
            ax.plot(x, y_smooth,
                    color=item["color"],
                    label=item["label"],
                    linewidth=2.0,
                    zorder=10)

        except Exception as e:
            print(f"处理 {item['label']} 失败: {e}")

    # ================= 2. 坐标轴与细节美化 =================

    # 设定坐标轴标签 (使用 LaTeX 公式使 Reward 更具学术感)
    plt.xlabel('训练回合数', fontsize=14, fontweight='bold')
    plt.ylabel('平均回合奖励', fontsize=14, fontweight='bold')

    # 去除顶部和右侧边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 限制坐标轴范围，防止两头留白过多
    plt.xlim(0, 60000)

    # 设置网格，只保留 Y 轴主网格，减少视觉干扰
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.3)
    ax.xaxis.grid(False)

    # 图例：放在右下角，去掉边框
    # plt.legend(loc='lower right', fontsize=12, frameon=False)
    # plt.legend(loc='lower right', bbox_to_anchor=(1, 0.11), fontsize=12, frameon=False)
    # 图例：放在右下角，利用 bbox_to_anchor 将其向上抬升
    # plt.legend(loc='lower right', bbox_to_anchor=(1, 0.11), fontsize=12, frameon=False)
    plt.legend(loc='lower right', fontsize=12, frameon=False)
    # 紧凑布局
    plt.tight_layout()

    # ================= 3. 多格式保存 =================
    save_dir = "paper_plots"
    os.makedirs(save_dir, exist_ok=True)

    # 1. 保存为 PNG (用于查看)
    plt.savefig(f"{save_dir}/reward3.png", bbox_inches='tight')

    # 2. 保存为 PDF (用于 LaTeX 或专业排版)
    # plt.savefig(f"{save_dir}/reward2.pdf", bbox_inches='tight')

    # 3. 保存为 SVG (这是你需要的，直接插入 Word 效果极佳)
    # 加上了之前的 'path' 设置后，这个 SVG 里的文字就是形状，不会丢字
    plt.savefig(f"{save_dir}/reward3.svg", bbox_inches='tight')


# ================= 3. 数据路径配置 =================
files_config1 = [
    {"path": "training_results_end2_sac/training_data_20260303_010222.xlsx", "label": "End-to-End SAC", "color": COLORS["Base"]},
    {"path": "training_results_ppo/training_data_20260227_174154.xlsx", "label": "Residual PPO", "color": COLORS["color0"]},
    {"path": "training_results_sac/training_data_20260227_174455.xlsx", "label": "Residual SAC", "color": COLORS["color2"]},
    {"path": "training_results_GPO_TS_SE-SCSA/training_data_20260228_200954.xlsx", "label": "GRP-SAC", "color": COLORS["Best"]},
]
files_config2 = [
    {"path": "training_results_TS_SE-SCSA/training_data_20260303_182913.xlsx", "label": "GRP-SAC (w/o GPO)", "color": COLORS["color0"]},
    {"path": "training_results_GPO_SE-SCSA/training_data_20260303_152014.xlsx", "label": "GRP-SAC (w/o PKD)", "color": COLORS["color2"]},
    {"path": "training_results_GPO_TS_SE-SCSA/training_data_20260228_200954.xlsx", "label": "GRP-SAC", "color": COLORS["Best"]},
]
files_config3= [
    {"path": "training_results_GPO_TS_SCSA/training_data_20260305_184206.xlsx", "label": "GRP-SAC (Vanilla)", "color": COLORS["Base"]},
    {"path": "training_results_GPO_TS_SE-SCSA-w_o_LayerNorm/training_data_20260304_101017.xlsx", "label": "GRP-SAC (w/o Norm)", "color": COLORS["color0"]},
    {"path": "training_results_GPO_TS_SCSA-LayerNorm/training_data_20260306_105619.xlsx", "label": "GRP-SAC (w/o SE)", "color": COLORS["color2"]},
    {"path": "training_results_GPO_TS_SE-SCSA/training_data_20260228_200954.xlsx", "label": "GRP-SAC", "color": COLORS["Best"]},
]

if __name__ == "__main__":
    plot_academic_reward(files_config1)