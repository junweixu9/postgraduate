# import numpy as np
# import matplotlib.pyplot as plt
#
# # 设置随机种子以保证结果可重复
# np.random.seed(42)
#
# # 生成时间轴
# t = np.linspace(0, 1, 100)  # 1秒的时间范围，采样点数为1000
#
# # 生成随机信号
# frequency = 5  # 假设信号的频率为5 Hz
# noise_amplitude = 0.5  # 噪声的幅度
# signal = np.sin(2 * np.pi * frequency * t) + noise_amplitude * np.random.normal(0, 1, len(t))
#
# # 绘制信号波形图
# plt.figure(figsize=(10, 4))
# plt.plot(t, signal,color='red')  # 将波形颜色设置为黑色
# plt.title('Randomly Generated Signal')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# # plt.grid(True)  # 去掉网格线
# plt.legend()
# plt.show()



def generate_image(text="程序已跑完", output_path="completed_matplotlib.png"):
    # 设置图像参数
    fig = plt.figure(figsize=(6, 3), dpi=300)  # 图像尺寸（宽600px，高300px）
    ax = plt.axes([0, 0, 1, 1], frameon=False)  # 全屏显示，无边框
    plt.axis('off')  # 隐藏坐标轴

    # 设置背景颜色（白色）
    ax.set_facecolor('white')

    font_path = 'simhei.ttf'  # 如果使用默认字体可删除此行和下面的 FontProperties

    # 自定义字体（中文字体需要指定路径）
    font_properties = {
        'family': 'SimHei',  # 中文字体名称（需系统支持）
        'size': 24,  # 字体大小
        'color': 'black',  # 字体颜色
        'weight': 'bold'  # 粗体
    }

    # 如果需要指定字体路径，使用FontProperties
    # from matplotlib.font_manager import FontProperties
    # prop = FontProperties(fname=font_path)
    # font_properties['fontproperties'] = prop

    # 添加文本（居中显示）
    ax.text(
        0.5,  # x坐标（0-1比例）
        0.5,  # y坐标（0-1比例）
        text,
        ha='center',  # 水平居中
        va='center',  # 垂直居中
        **font_properties
    )

    plt.show()  # 显示图形



import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties # 导入字体管理器

if __name__ == "__main__":
    # --- 中文字体设置 (关键步骤) ---
    font_to_try = ['SimSun', 'Microsoft YaHei', 'SimHei', 'KaiTi', 'FangSong', 'Arial Unicode MS', 'sans-serif']
    font_found = False
    for font_name in font_to_try:
        try:
            plt.rcParams['font.sans-serif'] = [font_name] # 设置一个包含中文的字体
            plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
            # 简单测试一下字体是否真的能用 (可选，但有助于调试)
            fig_test, ax_test = plt.subplots(figsize=(1,1))
            ax_test.text(0.5, 0.5, "测试中文类", fontproperties=FontProperties(fname=None, family=font_name)) # 直接用family
            plt.close(fig_test) # 关闭测试图
            print(f"尝试使用字体: '{font_name}' 成功。")
            font_found = True
            break # 找到可用字体后即跳出循环
        except Exception as e:
            print(f"尝试使用字体: '{font_name}' 失败。错误: {e}")
            plt.rcParams['font.sans-serif'] = ['sans-serif'] # 回退到通用无衬线字体

    if not font_found:
        print("\n警告：未能成功加载指定的中文字体。图表中的中文可能无法正确显示。")
        print("请确保您的系统已安装至少一种中文字体 (如 SimSun, Microsoft YaHei, SimHei)，")
        print("并且 Matplotlib 可以访问到它。")
        print("您可能需要清除 Matplotlib 的字体缓存并重启 Python 环境。")
        print("清除缓存通常在用户目录下的 .matplotlib 文件夹中删除 fontList.json 或 fontlist-v3xx.json。")
        # 不建议在生产代码中自动删除，但可以提示用户手动操作
        # import os
        # import matplotlib
        # try:
        #     font_cache_dir = matplotlib.get_cachedir()
        #     for f in os.listdir(font_cache_dir):
        #         if f.startswith('fontlist') and (f.endswith('.json') or f.endswith('.cache')):
        #             os.remove(os.path.join(font_cache_dir, f))
        #             print(f"已删除字体缓存文件: {f}")
        #     print("请重启您的Python脚本/环境使字体缓存重建生效。")
        # except Exception as e_cache:
        #     print(f"尝试删除字体缓存时出错: {e_cache}")


    # --- 数据定义 ---
    x = np.arange(0, 9)
    y_head = [0.285, 0.208, 0.20]
    y_tail = [0.20, 0.09, 0.08, 0.03, 0.02, 0.0075, 0.005]

    # --- 图表美化 ---
    plt.style.use('seaborn-v0_8-whitegrid')
    matplotlib.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(figsize=(12, 7))

    color_head = '#8E44AD'
    color_tail = '#3498DB'
    edge_color = '#2C3E50'

    ax.fill_between(x[0:3], y_head, color=color_head, edgecolor=edge_color, linewidth=1.5, alpha=0.85, label='头部区域')
    ax.fill_between(x[2:9], y_tail, color=color_tail, edgecolor=edge_color, linewidth=1.5, alpha=0.85, label='尾部区域')

    ax.set_xlabel('HW标签类别', fontsize=16, fontweight='bold', labelpad=15)
    ax.set_xticks(x)
    xtick_labels_original = [4, 5, 3, 6, 2, 7, 1, 8, 0]
    xtick_labels_descriptive = [f'类别 {label}' for label in xtick_labels_original] # “类”字在这里使用
    ax.set_xticklabels(xtick_labels_descriptive, fontsize=12, rotation=45, ha="right")

    ax.set_ylabel('样本数量占比', fontsize=16, fontweight='bold', labelpad=15)
    ax.set_title('HW标签类别长尾分布图', fontsize=20, fontweight='bold', pad=20)

    legend = ax.legend(fontsize=14, title='数据区域', title_fontsize='15', frameon=True, fancybox=True, shadow=True, borderpad=1, facecolor='white', framealpha=0.9)
    if legend.get_title(): # 确保图例标题存在
         legend.get_title().set_fontweight('bold')


    ax.grid(True, linestyle=':', linewidth='0.8', color='gray', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['left'].set_color('dimgray')
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['bottom'].set_color('dimgray')

    ax.tick_params(axis='both', which='major', colors='dimgray', labelsize=12)

    # plt.tight_layout() 应该在所有绘图元素都添加完毕后调用
    # 如果仍然出现字体警告，并且中文显示为方框，则问题仍在字体配置
    try:
        plt.tight_layout()
    except Exception as e_layout:
        print(f"调用 plt.tight_layout() 时发生错误: {e_layout}")


    # plt.savefig("beautified_long_tail_plot_v3.png", dpi=300, bbox_inches='tight')
    plt.show()