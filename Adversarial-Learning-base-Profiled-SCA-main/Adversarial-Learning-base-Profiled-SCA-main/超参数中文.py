import matplotlib

import matplotlib.pyplot as plt
from mksci_font import config_font

# 一行代码完成配置
config_font({"font.size": 24})  # 可同时设置字号

# 其余绘图代码保持不变...

# 数据准备
x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5]
ASCAD_gaussian_4 = [3953,3294, 2438, 2787, 2333, 1675, 1503, 1419, 1379.6, 1154.4, 902.2, 1049.4, 987.2, 1097.2, 1208.6, 1241.4]
XMEAGA = [5000, 33.8, 25.2, 30.2, 32.8, 31.2, 25.8, 30, 26.8, 24.4, 21, 23.6, 25.2, 27.4, 25.8, 25.4]
SAKURA_AES = [5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 3681, 3544, 1436, 3987, 5000, 5000, 5000, 5000]

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制曲线
plt.plot(x, ASCAD_gaussian_4, marker='o', linestyle='-', color='black', label='ASCAD(无噪声→四级别噪声)')
# plt.plot(x, XMEAGA, marker='s', linestyle='--', color='black', label='XMEGA(设备 1→ 设备 2)')
plt.plot(x, SAKURA_AES, marker='^', linestyle=':', color='black', label='SAKURA_AES(设备 1→ 设备 2)')

# 坐标轴范围
plt.xlim(0, 1.6)
plt.ylim(0, 7000)

# 刻度、标签、图例字体大小（字体已由 rcParams 统一控制）
plt.tick_params(axis='both', which='major', labelsize=20)
plt.xlabel('超参数大小', fontsize=20)
plt.ylabel('攻击轨迹数', fontsize=20)
plt.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize=20)
plt.grid(True, linestyle='--', alpha=0.7)

# 添加红色虚线框
rect = matplotlib.patches.Rectangle(
    (0.75, 650), 0.5, 4000,
    edgecolor='red', facecolor='none', linestyle='--', linewidth=2
)
plt.gca().add_patch(rect)

# 保存和显示图形
plt.savefig("实际攻击SAKURA&ASCAD.png", bbox_inches='tight', dpi=140, pad_inches=0.1)
print("图表已保存为 hyperparameter_impact_large_font_chinese.png")
plt.show()
# ret = requests.get('https://api.day.app/Cqdu3932Dcbe3dduH2EJPM/训练/完成')