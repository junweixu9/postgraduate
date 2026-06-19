import os
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# =========================
# 0) 全局设置
# =========================
torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def seed_everything(seed: int = 8):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# =========================
# 1) Dataset & DataLoader
# =========================
class TorchDataset(Dataset):
    def __init__(self, trs_file, label_file, trace_num, trace_offset, trace_length):
        self.trs_file = trs_file
        self.label_file = label_file
        self.trace_num = trace_num
        self.trace_offset = trace_offset
        self.trace_length = trace_length

    def __getitem__(self, i):
        idx = i % self.trace_num
        trace = self.trs_file[idx, self.trace_offset: self.trace_offset + self.trace_length]
        x = torch.from_numpy(trace).float().unsqueeze(0)  # (1, L)
        y = torch.tensor(self.label_file[idx], dtype=torch.long)
        return x, y

    def __len__(self):
        return self.trace_num

def make_loader(batch_size, kwargs, train: bool):
    ds = TorchDataset(**kwargs)
    num_workers = 4 if train else 2
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        drop_last=train,            # test/valid 不 drop_last，避免丢数据
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

# =========================
# 2) AES 常量（GE 需要）
# =========================
Sbox = [99, 124, 119, 123, 242, 107, 111, 197, 48, 1, 103, 43, 254, 215, 171, 118, 202, 130, 201, 125, 250, 89, 71,
        240, 173, 212, 162, 175, 156, 164, 114, 192, 183, 253, 147, 38, 54, 63, 247, 204, 52, 165, 229, 241, 113, 216,
        49, 21, 4, 199, 35, 195, 24, 150, 5, 154, 7, 18, 128, 226, 235, 39, 178, 117, 9, 131, 44, 26, 27, 110, 90, 160,
        82, 59, 214, 179, 41, 227, 47, 132, 83, 209, 0, 237, 32, 252, 177, 91, 106, 203, 190, 57, 74, 76, 88, 207, 208,
        239, 170, 251, 67, 77, 51, 133, 69, 249, 2, 127, 80, 60, 159, 168, 81, 163, 64, 143, 146, 157, 56, 245, 188,
        182, 218, 33, 16, 255, 243, 210, 205, 12, 19, 236, 95, 151, 68, 23, 196, 167, 126, 61, 100, 93, 25, 115, 96,
        129, 79, 220, 34, 42, 144, 136, 70, 238, 184, 20, 222, 94, 11, 219, 224, 50, 58, 10, 73, 6, 36, 92, 194, 211,
        172, 98, 145, 149, 228, 121, 231, 200, 55, 109, 141, 213, 78, 169, 108, 86, 244, 234, 101, 122, 174, 8, 186,
        120, 37, 46, 28, 166, 180, 198, 232, 221, 116, 31, 75, 189, 139, 138, 112, 62, 181, 102, 72, 3, 246, 14, 97,
        53, 87, 185, 134, 193, 29, 158, 225, 248, 152, 17, 105, 217, 142, 148, 155, 30, 135, 233, 206, 85, 40, 223, 140,
        161, 137, 13, 191, 230, 66, 104, 65, 153, 45, 15, 176, 84, 187, 22]

HW_byte = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2,
           3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2, 3,
           3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3,
           4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4,
           3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5,
           6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4,
           4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5,
           6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8]

Sbox_np = np.array(Sbox, dtype=np.int32)
HW_byte_np = np.array(HW_byte, dtype=np.int32)

def calculate_HW(data):
    return HW_byte_np[data.astype(int)]

# =========================
# 3) MMD + CORAL
# =========================
def gaussian_kernel_matrix(x, y, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    total = torch.cat([x, y], dim=0)
    dist2 = torch.cdist(total, total, p=2) ** 2

    n_samples = total.size(0)
    if fix_sigma is None:
        sigma2 = dist2.sum() / (n_samples * n_samples - n_samples)
    else:
        sigma2 = torch.tensor(fix_sigma, device=total.device, dtype=total.dtype)

    sigma2 = sigma2 / (kernel_mul ** (kernel_num // 2))
    sigmas = [sigma2 * (kernel_mul ** i) for i in range(kernel_num)]
    return sum(torch.exp(-dist2 / s) for s in sigmas)

def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    bs = min(source.size(0), target.size(0))
    source = source[:bs]
    target = target[:bs]
    K = gaussian_kernel_matrix(source, target, kernel_mul, kernel_num, fix_sigma)
    XX = K[:bs, :bs]
    YY = K[bs:, bs:]
    XY = K[:bs, bs:]
    YX = K[bs:, :bs]
    return (XX + YY - XY - YX).mean()

def _covariance(feat: torch.Tensor) -> torch.Tensor:
    n, d = feat.size(0), feat.size(1)
    if n <= 1:
        return torch.zeros((d, d), device=feat.device, dtype=feat.dtype)
    feat = feat - feat.mean(dim=0, keepdim=True)
    return feat.t().mm(feat) / (n - 1)

def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Deep CORAL: (1/(4 d^2)) * ||Cs - Ct||_F^2
    """
    bs = min(source.size(0), target.size(0))
    source = source[:bs]
    target = target[:bs]
    d = source.size(1)
    Cs = _covariance(source)
    Ct = _covariance(target)
    return (Cs - Ct).pow(2).sum() / (4.0 * (d ** 2))

# =========================
# 4) 模型：训练用 forward(source,target)，测试用 forward_single(x)
# =========================
class CDP_Net(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=1),
            nn.SELU(),
            nn.BatchNorm1d(32),
            nn.AvgPool1d(kernel_size=2, stride=2),

            nn.Conv1d(32, 64, kernel_size=50),
            nn.SELU(),
            nn.BatchNorm1d(64),
            nn.AvgPool1d(kernel_size=50, stride=50),

            nn.Conv1d(64, 128, kernel_size=3),
            nn.SELU(),
            nn.BatchNorm1d(128),
            nn.AvgPool1d(kernel_size=2, stride=2),

            nn.Flatten()
        )
        self.classifier_1 = nn.Sequential(nn.Linear(256, 20), nn.SELU())
        self.classifier_2 = nn.Sequential(nn.Linear(20, 20), nn.SELU())
        self.classifier_3 = nn.Sequential(nn.Linear(20, 20), nn.SELU())
        self.final_classifier = nn.Sequential(nn.Linear(20, num_classes))

    def encode(self, x):
        z0 = self.features(x)          # (bs, 256)
        z1 = self.classifier_1(z0)     # (bs, 20)
        z2 = self.classifier_2(z1)     # (bs, 20)
        z3 = self.classifier_3(z2)     # (bs, 20)
        return z0, z1, z2, z3

    def forward(self, source, target):
        s0, s1, s2, s3 = self.encode(source)
        logits = self.final_classifier(s3)

        t0, t1, t2, _ = self.encode(target)

        mmd = mmd_rbf(s0, t0) + mmd_rbf(s1, t1) + mmd_rbf(s2, t2)
        cor = coral_loss(s0, t0) + coral_loss(s1, t1) + coral_loss(s2, t2)
        return logits, mmd, cor

    @torch.no_grad()
    def forward_single(self, x):
        _, _, _, z3 = self.encode(x)
        return self.final_classifier(z3)

# =========================
# 5) 时钟抖动（你原逻辑）
# =========================
def regulateMatrix(M, size):
    Z = np.zeros((len(M), size), dtype=np.float32)
    for enu, row in enumerate(M):
        row = np.asarray(row, dtype=np.float32)
        if len(row) <= size:
            Z[enu, :len(row)] = row
        else:
            Z[enu, :] = row[:size]
    return Z

def addClockJitter(traces, clock_range, trace_length):
    print('Add clock jitters...')
    output_traces = []
    for trace_idx in range(len(traces)):
        if trace_idx % 2000 == 0:
            print(f'{trace_idx}/{len(traces)}')
        trace = traces[trace_idx]
        point = 0
        new_trace = []
        while point < len(trace) - 1:
            new_trace.append(int(trace[point]))
            r = random.randint(-clock_range, clock_range)
            if r <= 0:
                point += abs(r)
            else:
                avg_point = int((int(trace[point]) + int(trace[point + 1])) / 2)
                new_trace.extend([avg_point] * r)
            point += 1
        output_traces.append(new_trace)
    return regulateMatrix(output_traces, trace_length)

# =========================
# 6) 评估：Accuracy/Confusion + GE/SR
# =========================
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def plot_guessing_entropy(preds, plaintext, real_key, labeling_method='hw'):
    """
    preds: (N, C) softmax 概率
    plaintext: (N,) 选择的 attack byte 明文
    real_key: int in [0,255]
    """
    num_averaged = 100
    trace_num_max = 5000
    key_guesses = np.arange(256, dtype=np.int32)

    N = len(plaintext)
    trace_num_max = min(trace_num_max, N)

    guessing_entropy = np.zeros((num_averaged, trace_num_max))
    success_flag = np.zeros((num_averaged, trace_num_max))

    for t in range(num_averaged):
        idx = np.random.choice(N, trace_num_max, replace=False)
        pt = plaintext[idx]                 # (T,)
        P = preds[idx]                      # (T, C)

        state = pt[:, None] ^ key_guesses[None, :]   # (T, 256)
        sbox_out = Sbox_np[state]                    # (T, 256)

        if labeling_method == 'identity':
            labels = sbox_out                        # (T, 256) in [0,255]
        else:
            labels = HW_byte_np[sbox_out]            # (T, 256) in [0,8]

        row = np.arange(trace_num_max)[:, None]
        probs = P[row, labels]                       # (T, 256)
        log_probs = np.log(probs + 1e-40)
        scores = np.cumsum(log_probs, axis=0)        # (T, 256)

        real_scores = scores[:, real_key][:, None]
        ranks = np.sum(scores > real_scores, axis=1)

        guessing_entropy[t, :] = ranks
        success_flag[t, :] = (ranks == 0).astype(int)

    avg_ge = guessing_entropy.mean(axis=0)
    avg_sr = success_flag.mean(axis=0)

    converge = np.where(avg_ge < 1)[0]
    if len(converge) == 0:
        print(f"GE did not converge to < 1 within {trace_num_max} traces.")
    else:
        print(f"Traces to reach GE < 1: {converge[0] + 1}")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(avg_ge)
    plt.xlabel('Number of traces')
    plt.ylabel('Guessing Entropy')

    plt.subplot(1, 2, 2)
    plt.plot(avg_sr)
    plt.xlabel('Number of traces')
    plt.ylabel('Success Rate')
    plt.show()

@torch.no_grad()
def test_one_domain(model: CDP_Net, loader, device, class_num, real_key, plaintext_byte,
                    labeling_method='hw', show_cm=True, show_ge=True, name=''):
    model.eval()
    softmax = nn.Softmax(dim=1)

    all_probs = []
    all_pred = []
    all_true = []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model.forward_single(xb)
        prob = softmax(logits).cpu().numpy()

        pred = logits.argmax(dim=1).cpu().numpy()
        true = yb.cpu().numpy()

        all_probs.append(prob)
        all_pred.append(pred)
        all_true.append(true)

    probs = np.concatenate(all_probs, axis=0)  # (N, C)
    pred = np.concatenate(all_pred, axis=0)
    true = np.concatenate(all_true, axis=0)

    acc = (pred == true).mean() * 100.0
    print(f'[{name}] Accuracy: {acc:.2f}%  (N={len(true)})')

    if show_cm:
        cm = confusion_matrix(true, pred)
        plot_confusion_matrix(cm, title=f'CM - {name}')

    if show_ge:
        plot_guessing_entropy(probs, plaintext_byte[:len(probs)], real_key, labeling_method=labeling_method)

    return acc

@torch.no_grad()
def report_alignment_mmd_coral(model: CDP_Net, source_loader, target_loader, device, iters=50):
    """
    仅用于报告：在测试阶段估计平均 MMD / CORAL（不影响预测）
    """
    model.eval()
    it_s = iter(source_loader)
    it_t = iter(target_loader)

    mmd_vals, cor_vals = [], []
    for k in range(iters):
        try:
            sx, _ = next(it_s)
        except StopIteration:
            it_s = iter(source_loader)
            sx, _ = next(it_s)

        try:
            tx, _ = next(it_t)
        except StopIteration:
            it_t = iter(target_loader)
            tx, _ = next(it_t)

        sx = sx.to(device, non_blocking=True)
        tx = tx.to(device, non_blocking=True)

        _, mmd, cor = model(sx, tx)
        mmd_vals.append(float(mmd.detach().cpu()))
        cor_vals.append(float(cor.detach().cpu()))

    print(f'[Alignment] Avg MMD={np.mean(mmd_vals):.6f} | Avg CORAL={np.mean(cor_vals):.6f} over {iters} iters')

# =========================
# 7) main：加载 fine-tuned MMD-CORAL 模型并测试
# =========================
if __name__ == '__main__':
    # ----------- 配置 -----------
    seed_everything(8)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    source_device_id = 0
    target_device_id = 1
    real_key_01 = 224
    real_key_02 = 224

    labeling_method = 'hw'   # 'identity' or 'hw'
    class_num = 9 if labeling_method == 'hw' else 256

    batch_size = 200
    trace_offset = 0
    trace_length = 700

    countermeasure = '_clockjitter_level1'
    clock_range = 1
    source_file_path = './Data/ASCAD/'

    # 这里要与训练时一致
    adjust_flag = True
    flag = "real" if adjust_flag else "fake"

    # ✅ 改成你保存的 fine-tuned MMD-CORAL checkpoint 路径
    ckpt_path = f'./models/{countermeasure}_{flag}_best_valid_loss_fine_tuned_MMD_CORAL_device{source_device_id}_to_{target_device_id}.pth'
    print('Loading ckpt:', ckpt_path)

    # ----------- 读数据（测试集）-----------
    X_attack_source = np.load(source_file_path + 'X_attack.npy')
    Y_attack_source = np.load(source_file_path + 'Y_attack.npy')

    if labeling_method == 'hw':
        Y_attack_source = calculate_HW(Y_attack_source)

    # target：对 attack 做 clock jitter，然后按你的逻辑切分（前 target_finetune_num 用于适配，其后用于测试）
    target_finetune_num = 200
    target_test_num = 9000
    source_test_num = 10000

    X_attack_target = addClockJitter(X_attack_source, clock_range, trace_length)
    Y_attack_target = Y_attack_source

    # 明文（攻击 byte = 2）
    plaintexts = np.load(source_file_path + 'plaintexts_attack.npy')  # (N,16)
    plaintexts_source = plaintexts[:source_test_num, 2]
    plaintexts_target = plaintexts[target_finetune_num: target_finetune_num + target_test_num, 2]

    kwargs_source_test = dict(
        trs_file=X_attack_source[:source_test_num, :],
        label_file=Y_attack_source[:source_test_num],
        trace_num=source_test_num,
        trace_offset=trace_offset,
        trace_length=trace_length,
    )
    kwargs_target_test = dict(
        trs_file=X_attack_target[target_finetune_num: target_finetune_num + target_test_num, :],
        label_file=Y_attack_target[target_finetune_num: target_finetune_num + target_test_num],
        trace_num=target_test_num,
        trace_offset=trace_offset,
        trace_length=trace_length,
    )

    source_test_loader = make_loader(batch_size, kwargs_source_test, train=False)
    target_test_loader = make_loader(batch_size, kwargs_target_test, train=False)
    print('Load test data complete!')

    # ----------- 构建模型并加载权重 -----------
    model = CDP_Net(num_classes=class_num).to(device)
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        sd = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(sd, strict=True)
        print('Checkpoint loaded.')
    else:
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    # ----------- 可选：报告对齐指标（MMD/CORAL）-----------
    report_alignment_mmd_coral(model, source_test_loader, target_test_loader, device, iters=50)

    # ----------- 测试：Source / Target -----------
    print('\n=== Result on SOURCE device ===')
    test_one_domain(
        model=model,
        loader=source_test_loader,
        device=device,
        class_num=class_num,
        real_key=real_key_01,
        plaintext_byte=plaintexts_source,
        labeling_method=labeling_method,
        show_cm=True,
        show_ge=True,
        name='SOURCE'
    )

    print('\n=== Result on TARGET device ===')
    test_one_domain(
        model=model,
        loader=target_test_loader,
        device=device,
        class_num=class_num,
        real_key=real_key_02,
        plaintext_byte=plaintexts_target,
        labeling_method=labeling_method,
        show_cm=True,
        show_ge=True,
        name='TARGET'
    )
