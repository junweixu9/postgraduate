import numpy as np

# =========================
# 1) 统计分布 + KL / JS
# =========================
def label_distribution(y: np.ndarray, class_num: int, eps: float = 1e-12) -> np.ndarray:
    """
    y: 1D int labels
    class_num: number of classes
    eps: Laplace-like smoothing for stability
    """
    y = y.astype(np.int64).ravel()
    counts = np.bincount(y, minlength=class_num).astype(np.float64)
    p = counts / max(counts.sum(), 1.0)
    # smoothing to avoid zeros (important for KL)
    p = np.clip(p, eps, 1.0)
    p = p / p.sum()
    return p

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q)"""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return float(np.sum(p * np.log(p / q)))

def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """JS(p, q) = 0.5*KL(p||m) + 0.5*KL(q||m)"""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

def summarize_shift(y_source: np.ndarray, y_target: np.ndarray, class_num: int, name: str = ""):
    ps = label_distribution(y_source, class_num)
    pt = label_distribution(y_target, class_num)
    kl_t_s = kl_divergence(pt, ps)  # KL(pt||ps)
    kl_s_t = kl_divergence(ps, pt)  # KL(ps||pt)
    js = js_divergence(ps, pt)

    print(f"\n===== Label Shift ({name}) =====")
    print(f"class_num = {class_num}")
    print(f"KL(pt||ps) = {kl_t_s:.6f}")
    print(f"KL(ps||pt) = {kl_s_t:.6f}")
    print(f"JS(ps,pt)  = {js:.6f}")
    print(f"ps(y) top-10: idx/prob =", sorted(list(enumerate(ps)), key=lambda x: -x[1])[:10])
    print(f"pt(y) top-10: idx/prob =", sorted(list(enumerate(pt)), key=lambda x: -x[1])[:10])
    return ps, pt, kl_t_s, kl_s_t, js


# =========================
# 2) 你的数据加载后：分别算 ID 与 HW
# =========================
source_file_path = './Data/device1/'
target_file_path = './Data/device2/'

# 你原始代码（示例）：
X_train_source = np.load(source_file_path + 'X_train.npy')
Y_train_source = np.load(source_file_path + 'Y_train.npy')
X_attack_target = np.load(target_file_path + 'X_attack.npy')
Y_attack_target = np.load(target_file_path + 'Y_attack.npy')

target_finetune_num = 200
# ---- A) ID label shift（256类）----
# 注意：这里要用“原始ID标签”，所以先保留一份
Y_train_source_id = Y_train_source.copy()
Y_attack_fine_id = Y_attack_target[0:target_finetune_num].copy()
Y_attack_target_id = Y_attack_target[target_finetune_num:].copy()

summarize_shift(Y_attack_fine_id, Y_attack_target_id, class_num=256, name="ID: source_train vs target_attack")

HW_byte = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2,
           3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2, 3,
           3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3,
           4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4,
           3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5,
           6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4,
           4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5,
           6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8]

HW_byte_np = np.array(HW_byte, dtype=np.int32)
def calculate_HW(data):
    # 【优化】使用向量化索引代替列表推导式，速度极快
    return HW_byte_np[data.astype(int)]

# ---- B) HW label shift（9类）----
# 你已有 calculate_HW() 就直接用
Y_train_source_hw = calculate_HW(Y_train_source_id)
Y_attack_fine_hw = calculate_HW(Y_attack_target_id[0:target_finetune_num])
Y_attack_target_hw = calculate_HW(Y_attack_target_id[target_finetune_num:])

summarize_shift(Y_attack_fine_hw, Y_attack_target_hw, class_num=9, name="HW: source_train vs target_attack")