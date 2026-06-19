import os
import random
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

# =========================
# 0) 全局加速/复现设置
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
        drop_last=train,              # train=True 才 drop_last，valid/test 不 drop_last
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

# =========================
# 2) HW
# =========================
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
    return HW_byte_np[data.astype(int)]

# =========================
# 3) Logit Adjustment
# =========================
def compute_adjustment_1(Y_profiling, tro, classes=9, eps=1e-12):
    counts = np.bincount(Y_profiling, minlength=classes).astype(np.float64)
    pi = counts / np.maximum(counts.sum(), 1.0)
    adj = np.log(np.power(pi, tro) + eps)   # log(pi^tau)
    return adj.astype(np.float32)

# =========================
# 4) MMD (multi-kernel RBF)
# =========================
def gaussian_kernel_matrix(x, y, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    total = torch.cat([x, y], dim=0)  # (n+m, d)
    dist2 = torch.cdist(total, total, p=2) ** 2

    n_samples = total.size(0)
    if fix_sigma is None:
        sigma2 = dist2.sum() / (n_samples * n_samples - n_samples)
    else:
        sigma2 = torch.tensor(fix_sigma, device=total.device, dtype=total.dtype)

    sigma2 = sigma2 / (kernel_mul ** (kernel_num // 2))
    sigmas = [sigma2 * (kernel_mul ** i) for i in range(kernel_num)]
    kernels = [torch.exp(-dist2 / s) for s in sigmas]
    return sum(kernels)

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

# =========================
# 4.5) CORAL loss (Deep CORAL)
# =========================
def _covariance(feat: torch.Tensor) -> torch.Tensor:
    """
    feat: (n, d)
    return: (d, d) covariance
    """
    n = feat.size(0)
    d = feat.size(1)
    # 避免 n=1 的极端情况
    if n <= 1:
        return torch.zeros((d, d), device=feat.device, dtype=feat.dtype)

    feat = feat - feat.mean(dim=0, keepdim=True)
    # unbiased: / (n-1)
    cov = feat.t().mm(feat) / (n - 1)
    return cov

def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    source/target: (n, d)
    Deep CORAL: (1/(4 d^2)) * ||C_s - C_t||_F^2
    常见实现按 Sun & Saenko (2016) 的归一化形式写法。:contentReference[oaicite:2]{index=2}
    """
    bs = min(source.size(0), target.size(0))
    source = source[:bs]
    target = target[:bs]
    d = source.size(1)

    Cs = _covariance(source)
    Ct = _covariance(target)
    loss = (Cs - Ct).pow(2).sum() / (4.0 * (d ** 2))
    return loss

# =========================
# 5) 模型：多层 MMD + 多层 CORAL（MMD-CORAL / MC）
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

    def forward(self, source, target):
        # source
        s0 = self.features(source)          # (bs, 256)
        s1 = self.classifier_1(s0)          # (bs, 20)
        s2 = self.classifier_2(s1)          # (bs, 20)
        s3 = self.classifier_3(s2)          # (bs, 20)
        logits = self.final_classifier(s3)  # (bs, C)

        # target
        t0 = self.features(target)

        # ===== MMD: 你原来的多层对齐 =====
        mmd = mmd_rbf(s0, t0)
        t1 = self.classifier_1(t0); mmd = mmd + mmd_rbf(s1, t1)
        t2 = self.classifier_2(t1); mmd = mmd + mmd_rbf(s2, t2)

        # ===== CORAL: 同层二阶协方差对齐 =====
        cor = coral_loss(s0, t0) + coral_loss(s1, t1) + coral_loss(s2, t2)

        return logits, mmd, cor

# =========================
# 6) 时钟抖动（保持你原函数）
# =========================
def regulateMatrix(M, size):
    maxlen = size
    Z = np.zeros((len(M), maxlen))
    for enu, row in enumerate(M):
        if len(row) <= maxlen:
            Z[enu, :len(row)] += row
        else:
            Z[enu, :] += row[:maxlen]
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
                for _ in range(r):
                    new_trace.append(avg_point)
            point += 1
        output_traces.append(new_trace)
    return regulateMatrix(output_traces, trace_length)

# =========================
# 7) 训练/验证
# =========================
@torch.no_grad()
def build_target_bank(target_loader, device):
    xs = []
    for xb, _ in target_loader:
        xs.append(xb.to(device, non_blocking=True))
    return torch.cat(xs, dim=0)  # (N, 1, L)

def safe_load_optimizer_state(optimizer, checkpoint_opt_state):
    """
    ✅ 避免你之前遇到的：different number of parameter groups
    如果 param_groups 数量不一致，就跳过加载 optimizer_state_dict（一般 fine-tune 也不需要加载）。
    """
    try:
        if len(optimizer.param_groups) != len(checkpoint_opt_state.get("param_groups", [])):
            print(f"[Warn] Optimizer param_groups mismatch: "
                  f"new={len(optimizer.param_groups)} ckpt={len(checkpoint_opt_state.get('param_groups', []))}. "
                  f"Skip loading optimizer_state_dict.")
            return
        optimizer.load_state_dict(checkpoint_opt_state)
        print("[Info] Optimizer state loaded.")
    except Exception as e:
        print(f"[Warn] Failed to load optimizer_state_dict, skipped. Reason: {repr(e)}")

def cdp_train_one_epoch(model, optimizer, source_loader, target_bank,
                        adjustments, lambda_mmd, lambda_coral, device,
                        log_interval=50, use_amp=True, grad_clip=5.0):
    model.train()
    clf_criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    num_iter = len(source_loader)
    bs = source_loader.batch_size
    tgtN = target_bank.size(0)

    for i, (sx, sy) in enumerate(source_loader, start=1):
        sx = sx.to(device, non_blocking=True)
        sy = sy.to(device, non_blocking=True)

        start = ((i - 1) * bs) % max(tgtN - bs + 1, 1)
        tx = target_bank[start:start + bs]
        if tx.size(0) < sx.size(0):
            rep = (sx.size(0) + tx.size(0) - 1) // tx.size(0)
            tx = tx.repeat(rep, 1, 1)[:sx.size(0)]

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits, mmd_loss, coral = model(sx, tx)
            if adjustments is not None:
                logits = logits + adjustments  # (1,C) broadcast
            clf_loss = clf_criterion(logits, sy)

            # ===== MMD-CORAL 总损失 =====
            loss = clf_loss + lambda_mmd * mmd_loss + lambda_coral * coral

        scaler.scale(loss).backward()
        if grad_clip is not None and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if i % log_interval == 0:
            print(f'Iter {i}/{num_iter} | total {loss.item():.6f} '
                  f'clf {clf_loss.item():.6f} mmd {mmd_loss.item():.6f} coral {coral.item():.6f}')

@torch.no_grad()
def cdp_validate(model, source_valid_loader, target_bank,
                 adjustments, lambda_mmd, lambda_coral, device):
    model.eval()
    clf_criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_clf = 0.0
    total_mmd = 0.0
    total_coral = 0.0
    correct = 0
    total = 0

    bs = source_valid_loader.batch_size
    tgtN = target_bank.size(0)

    for i, (sx, sy) in enumerate(source_valid_loader, start=1):
        sx = sx.to(device, non_blocking=True)
        sy = sy.to(device, non_blocking=True)

        start = ((i - 1) * bs) % max(tgtN - bs + 1, 1)
        tx = target_bank[start:start + bs]
        if tx.size(0) < sx.size(0):
            rep = (sx.size(0) + tx.size(0) - 1) // tx.size(0)
            tx = tx.repeat(rep, 1, 1)[:sx.size(0)]

        logits, mmd_loss, coral = model(sx, tx)

        clf_loss = clf_criterion(logits, sy)
        loss = clf_loss + lambda_mmd * mmd_loss + lambda_coral * coral

        total_loss += loss.item()
        total_clf += clf_loss.item()
        total_mmd += mmd_loss.item()
        total_coral += coral.item()

        pred = logits.argmax(dim=1)
        correct += (pred == sy).sum().item()
        total += sy.size(0)

    n = len(source_valid_loader)
    print(f'Validation | total {total_loss/n:.4f} clf {total_clf/n:.4f} '
          f'mmd {total_mmd/n:.4f} coral {total_coral/n:.4f} '
          f'acc {correct}/{total} ({100.0*correct/total:.2f}%)')
    return total_loss / n

# =========================
# 8) main
# =========================
if __name__ == '__main__':
    # ---- 超参（你原设定 + 新增 lambda_coral）
    source_device_id = 0
    target_device_id = 1
    labeling_method = 'hw'

    lambda_mmd = 0.1
    lambda_coral = 0.1   # ✅ 新增：CORAL 权重（建议先与 lambda_mmd 同量级）

    batch_size = 200
    finetune_epoch = 15
    lr = 0.002
    log_interval = 50

    train_num = 45000
    valid_num = 5000
    target_finetune_num = 200

    trace_offset = 0
    trace_length = 700

    countermeasure = '_clockjitter_level1'
    clock_range = 1
    source_file_path = './Data/ASCAD/'

    seed = 8
    seed_everything(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # ---- 读数据
    X_train_source = np.load(source_file_path + 'X_train.npy')
    Y_train_source = np.load(source_file_path + 'Y_train.npy')
    X_attack_source = np.load(source_file_path + 'X_attack.npy')
    Y_attack_source = np.load(source_file_path + 'Y_attack.npy')

    # ---- 构造 target（时钟抖动）
    X_attack_target = addClockJitter(X_attack_source, clock_range, trace_length)
    Y_attack_target = Y_attack_source

    if labeling_method == 'hw':
        class_num = 9
        Y_train_source = calculate_HW(Y_train_source)
    else:
        class_num = 256

    # ---- loaders
    kwargs_source_train = dict(
        trs_file=X_train_source[0:train_num, :],
        label_file=Y_train_source[0:train_num],
        trace_num=train_num,
        trace_offset=trace_offset,
        trace_length=trace_length,
    )
    kwargs_source_valid = dict(
        trs_file=X_train_source[train_num:train_num + valid_num, :],
        label_file=Y_train_source[train_num:train_num + valid_num],
        trace_num=valid_num,
        trace_offset=trace_offset,
        trace_length=trace_length,
    )
    kwargs_target_finetune = dict(
        trs_file=X_attack_target[0:target_finetune_num, :],
        label_file=Y_attack_target[0:target_finetune_num],
        trace_num=target_finetune_num,
        trace_offset=trace_offset,
        trace_length=trace_length,
    )

    source_train_loader = make_loader(batch_size, kwargs_source_train, train=True)
    source_valid_loader = make_loader(batch_size, kwargs_source_valid, train=False)
    target_finetune_loader = make_loader(batch_size, kwargs_target_finetune, train=False)

    print('Load data complete!')

    # ---- adjustments
    adjust_flag = True
    adjustments = None
    if adjust_flag:
        adj_np = compute_adjustment_1(Y_train_source[0:train_num], tro=1, classes=class_num)
        adjustments = torch.from_numpy(adj_np).view(1, -1).to(device)

    flag = "real" if adjust_flag else "fake"

    # ---- model
    model = CDP_Net(num_classes=class_num).to(device)

    pretrained_path = f'./models/{countermeasure}_{flag}_pre-trained_cpda_device{source_device_id}.pth'
    print("Loading model:", pretrained_path)
    checkpoint = None
    if os.path.exists(pretrained_path):
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        sd = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(sd, strict=True)
    else:
        print(f"Warning: Pretrained model not found at {pretrained_path}")

    optimizer = optim.Adam([
        {'params': model.features.parameters()},
        {'params': model.classifier_1.parameters()},
        {'params': model.classifier_2.parameters()},
        {'params': model.classifier_3.parameters()},
        {'params': model.final_classifier.parameters()}
    ], lr=lr)

    # ✅ 安全加载 optimizer state（避免你之前 param_groups 报错）
    if checkpoint is not None and isinstance(checkpoint, dict) and 'optimizer_state_dict' in checkpoint:
        safe_load_optimizer_state(optimizer, checkpoint['optimizer_state_dict'])

    # ---- target bank（一次性构建，训练/验证共用）
    target_bank = build_target_bank(target_finetune_loader, device)
    print('Target bank:', tuple(target_bank.shape))

    best_loss = float('inf')
    os.makedirs('./models', exist_ok=True)

    for epoch in range(1, finetune_epoch + 1):
        print(f'\n===== Epoch {epoch}/{finetune_epoch} =====')
        cdp_train_one_epoch(
            model=model,
            optimizer=optimizer,
            source_loader=source_train_loader,
            target_bank=target_bank,
            adjustments=adjustments,
            lambda_mmd=lambda_mmd,
            lambda_coral=lambda_coral,
            device=device,
            log_interval=log_interval,
            use_amp=True,        # 可改 False
            grad_clip=5.0        # 可改 None
        )

        val_loss = cdp_validate(
            model=model,
            source_valid_loader=source_valid_loader,
            target_bank=target_bank,
            adjustments=adjustments,
            lambda_mmd=lambda_mmd,
            lambda_coral=lambda_coral,
            device=device
        )

        if val_loss < best_loss:
            best_loss = val_loss
            save_path = (
                f'./models/{countermeasure}_{flag}_best_valid_loss_fine_tuned_MMD_CORAL_'
                f'device{source_device_id}_to_{target_device_id}.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'lambda_mmd': lambda_mmd,
                'lambda_coral': lambda_coral,
            }, save_path)
            print('Saved best checkpoint to:', save_path)
