import matplotlib

import matplotlib.pyplot as plt
from mksci_font import config_font

# 一行代码完成配置
config_font({"font.size": 24})  # 可同时设置字号

# 其余绘图代码保持不变...

# 数据准备
x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
ascad_fixed = [1036, 1103, 793, 720, 634, 515, 366, 270, 210, 184, 169, 187, 230, 269, 278, 290, 320]
ascad_rand = [663, 607, 647, 550, 460, 484, 400, 350, 320, 277, 330, 375, 354, 386, 414, 434, 470]
ches_ctf = [529, 497, 446, 439, 434, 429, 425, 420, 270, 220, 177, 204, 217, 208, 228, 240, 279]

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制曲线
plt.plot(x, ascad_fixed, marker='o', linestyle='-', color='black', label='ASCAD_fixed')
plt.plot(x, ascad_rand, marker='s', linestyle='--', color='black', label='ASCAD_rand')
plt.plot(x, ches_ctf, marker='^', linestyle=':', color='black', label='CHES_CTF')

# 坐标轴范围
plt.xlim(0, 1.75)
plt.ylim(0, max(ascad_fixed) + 200)

# 刻度、标签、图例字体大小（字体已由 rcParams 统一控制）
plt.tick_params(axis='both', which='major', labelsize=20)
plt.xlabel('超参数大小', fontsize=20)
plt.ylabel('攻击轨迹数', fontsize=20)
plt.legend(loc='best', fontsize=20)
plt.grid(True, linestyle='--', alpha=0.7)

# 添加红色虚线框
rect = matplotlib.patches.Rectangle(
    (0.85, 140), 0.4, 260,
    edgecolor='red', facecolor='none', linestyle='--', linewidth=2
)
plt.gca().add_patch(rect)

# 保存和显示图形
plt.savefig("hyperparameter_impact_large_font_chinese.png", bbox_inches='tight', dpi=140, pad_inches=0.1)
print("图表已保存为 hyperparameter_impact_large_font_chinese.png")
plt.show()
# ret = requests.get('https://api.day.app/Cqdu3932Dcbe3dduH2EJPM/训练/完成')