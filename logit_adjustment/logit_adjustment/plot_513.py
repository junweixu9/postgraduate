import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

if __name__ == "__main__":
    # --- 关键步骤：指定中文字体文件的绝对路径 ---
    # !! 请确保这里的路径是您系统中一个真实有效的字体文件路径 !!
    font_path_chinese = 'C:/Windows/Fonts/simhei.ttf' # 示例：使用黑体
    # 例如: font_path_chinese = 'C:/Windows/Fonts/msyh.ttc' # 微软雅黑
    # 例如: font_path_chinese = 'C:/Windows/Fonts/msyhbd.ttc' # 微软雅黑粗体 (更适合放大)
    # 例如: font_path_chinese = '/System/Library/Fonts/PingFang.ttc' # macOS 苹方
    # 例如: font_path_chinese = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc' # Linux 文泉驿微米黑

    custom_font = None
    try:
        custom_font = FontProperties(fname=font_path_chinese)
        if custom_font:
            print(f"成功加载字体文件: {font_path_chinese}")
            plt.rcParams['font.sans-serif'] = [custom_font.get_name()]
            plt.rcParams['axes.unicode_minus'] = False
        else:
            print(f"警告：未能从路径加载字体: {font_path_chinese}")
    except Exception as e:
        print(f"错误：加载字体文件 '{font_path_chinese}' 失败: {e}")
        print("--- 请务必将 font_path_chinese 修改为系统中有效的字体文件路径 ---")

    if not custom_font:
        print("\n--- 由于未能加载指定的字体文件，中文可能无法正确显示。---")
        # return

    # --- 数据定义 ---
    x_data = np.arange(0, 9)
    y_head = [0.285, 0.208, 0.20]
    y_tail = [0.20, 0.09, 0.08, 0.03, 0.02, 0.0075, 0.005]

    # --- 图表美化 ---
    plt.style.use('seaborn-v0_8-whitegrid')
    # 1. 进一步减小 figsize 的宽度，高度可能需要略微调整以容纳更大的字体
    fig, ax = plt.subplots(figsize=(7.0, 5.4)) # 例如，宽度从 7.5 减到 7.0，高度略增

    color_head = '#8E44AD'
    color_tail = '#3498DB'
    edge_color = '#2C3E50'

    font_props_to_use = custom_font if custom_font else None

    ax.fill_between(x_data[0:3], y_head, color=color_head, edgecolor=edge_color, linewidth=1.5, alpha=0.85, label='头部区域')
    ax.fill_between(x_data[2:9], y_tail, color=color_tail, edgecolor=edge_color, linewidth=1.5, alpha=0.85, label='尾部区域')

    ax.margins(x=0.01, y=0.01) # 保持非常小的边距

    # 2. 增大坐标轴标签和标题的字体，同时保持pad较小
    ax.set_xlabel('HW标签类别', fontsize=16, fontweight='bold', labelpad=7, fontproperties=font_props_to_use)
    ax.set_xticks(x_data)
    xtick_labels_original = [4, 5, 3, 6, 2, 7, 1, 8, 0]
    xtick_labels_descriptive = [f'{label}' for label in xtick_labels_original]
    ax.set_xticklabels(xtick_labels_descriptive, rotation=0, ha="right")
    if font_props_to_use:
        for label in ax.get_xticklabels():
            label.set_fontproperties(font_props_to_use)
            label.set_fontsize(14) # 增大刻度标签字体

    ax.set_ylabel('样本数量占比', fontsize=16, fontweight='bold', labelpad=7, fontproperties=font_props_to_use)
    if font_props_to_use:
        for label in ax.get_yticklabels():
            label.set_fontproperties(font_props_to_use)
            label.set_fontsize(14) # 增大刻度标签字体
    else:
        ax.tick_params(axis='y', labelsize=14)

    # ax.set_title('HW标签类别长尾分布图', fontsize=19, fontweight='bold', pad=12, fontproperties=font_props_to_use)

    # 增大图例字体，同时保持内部间距相对紧凑以适应窄图
    legend = ax.legend(
        fontsize=26,         # 用户期望的图例字体大小
        title='数据区域',
        title_fontsize=30,   # 用户期望的图例标题字体大小
        markerscale=2.0,
        borderpad=0.7,       # 边框填充，略微减小以适应窄图
        labelspacing=0.6,    # 标签间距，略微减小
        handletextpad=0.7,   # 标记和文字间距，略微减小
        frameon=True,
        fancybox=True,
        shadow=True,
        facecolor='white',
        framealpha=0.9,
        prop=font_props_to_use
    )
    if legend.get_title() and font_props_to_use:
        legend.get_title().set_fontproperties(font_props_to_use)

    ax.grid(True, linestyle=':', linewidth='0.8', color='gray', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['left'].set_color('dimgray')
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['bottom'].set_color('dimgray')

    # --- 关键的留白调整 ---

    # 不使用 plt.tight_layout()，完全依赖 fig.subplots_adjust() 进行精细控制
    # plt.tight_layout()

    # 3. 使用 fig.subplots_adjust() 进行非常积极的边距压缩
    # 您需要仔细调整这些值，以在窄图中为较大的字体腾出空间
    # left 和 bottom 可能需要比之前更大一点，以容纳Y轴标签和旋转的X轴标签
    # right 和 top 可能需要更接近1，以最大限度利用空间
    fig.subplots_adjust(left=0.18, right=0.98, top=0.90, bottom=0.25)
    # 示例值解释：
    # left=0.18: 左边留出18%的宽度给Y轴标签和刻度
    # right=0.98: 右边绘图区在98%处结束，只留2%空白（如果图例在图内，这个可以大）
    # top=0.90: 顶部绘图区在90%处结束，留10%给标题（如果标题字大，可能需要减小此值或增大figsize高度）
    # bottom=0.25: 底部留出25%的高度给X轴标签和旋转的刻度（旋转标签需要较多垂直空间）

    plt.savefig("final_plot_narrower_larger_font.png", dpi=300, bbox_inches='tight')
    # plt.show()