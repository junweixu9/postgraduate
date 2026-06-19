import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torch import nn, optim

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
        trace_tensor = torch.from_numpy(trace).float().unsqueeze(0)  # [1, L]
        label_tensor = torch.tensor(self.label_file[idx], dtype=torch.long)
        return trace_tensor, label_tensor

    def __len__(self):
        return self.trace_num


def load_training(batch_size, kwargs):
    data = TorchDataset(**kwargs)
    loader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    return loader


def load_testing(batch_size, kwargs):
    data = TorchDataset(**kwargs)
    loader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    return loader


# =========================
# 2) AES HW Table
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
# 3) 论文 DCAN: CMMD + MI
# =========================
def entropy_from_probs(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # p: [B, C], already softmax
    p = p.clamp(min=eps, max=1.0)
    return -(p * p.log()).sum(dim=1)  # [B]

def mutual_information_loss(p_t: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    DCAN Eq.(10): L_MI = mean(H(p(y|x))) - H(mean(p(y|x)))
    """
    cond_ent = entropy_from_probs(p_t, eps=eps).mean()
    mean_p = p_t.mean(dim=0, keepdim=True)  # [1, C]
    marg_ent = entropy_from_probs(mean_p, eps=eps).squeeze(0)  # scalar
    return cond_ent - marg_ent

def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(labels, num_classes=num_classes).float()

def linear_kernel(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # A: [n, d], B: [m, d]
    return A @ B.t()

def rbf_kernel_multi_sigma(X: torch.Tensor, Y: torch.Tensor, sigmas=(0.1, 1.0, 10.0, 100.0, 1000.0)) -> torch.Tensor:
    """
    多核 RBF：论文实现里用 5 个固定带宽的高斯核平均（文中给出带宽列表）
    """
    # ||x-y||^2 = x^2 + y^2 - 2xy
    x_norm = (X ** 2).sum(dim=1, keepdim=True)  # [n,1]
    y_norm = (Y ** 2).sum(dim=1, keepdim=True)  # [m,1]
    dist2 = x_norm + y_norm.t() - 2.0 * (X @ Y.t())
    dist2 = dist2.clamp_min(0.0)

    K = 0.0
    for s in sigmas:
        K = K + torch.exp(-dist2 / (2.0 * (s ** 2)))
    return K / float(len(sigmas))

def cmmd_loss(
    z_s: torch.Tensor,
    y_s_oh: torch.Tensor,
    z_t: torch.Tensor,
    y_t_oh: torch.Tensor,
    reg: float = 1e-3,
    sigmas=(0.1, 1.0, 10.0, 100.0, 1000.0),
) -> torch.Tensor:
    """
    DCAN Eq.(7) 的可微 estimator:
    L_CMMD = Tr(Ls * inv(Ls+λI) * Ks * inv(Ls+λI)) + Tr(Lt*inv(Lt+λI)*Kt*inv(Lt+λI))
             -2 Tr(Lts*inv(Ls+λI) * Kst * inv(Lt+λI))
    这里:
      Ks = k(z_s,z_s), Kt = k(z_t,z_t), Kst = k(z_s,z_t)
      Ls = l(y_s,y_s), Lt = l(y_t,y_t), Lts = l(y_t,y_s)
    """
    n_s = z_s.size(0)
    n_t = z_t.size(0)
    if n_s < 2 or n_t < 2:
        return z_s.new_tensor(0.0)

    Ks = rbf_kernel_multi_sigma(z_s, z_s, sigmas=sigmas)
    Kt = rbf_kernel_multi_sigma(z_t, z_t, sigmas=sigmas)
    Kst = rbf_kernel_multi_sigma(z_s, z_t, sigmas=sigmas)

    # label kernels (linear)
    Ls = linear_kernel(y_s_oh, y_s_oh)
    Lt = linear_kernel(y_t_oh, y_t_oh)
    Lts = linear_kernel(y_t_oh, y_s_oh)  # [n_t, n_s]

    I_s = torch.eye(n_s, device=z_s.device, dtype=z_s.dtype)
    I_t = torch.eye(n_t, device=z_t.device, dtype=z_t.dtype)

    Ls_tilde = Ls + reg * I_s
    Lt_tilde = Lt + reg * I_t

    # inv via solve (more stable than explicit inverse)
    inv_Ls = torch.linalg.solve(Ls_tilde, I_s)
    inv_Lt = torch.linalg.solve(Lt_tilde, I_t)

    Gs = inv_Ls @ Ls @ inv_Ls
    Gt = inv_Lt @ Lt @ inv_Lt
    Gts = inv_Lt @ Lts @ inv_Ls  # [n_t, n_s]

    term1 = torch.trace(Gs @ Ks)
    term2 = torch.trace(Gt @ Kt)
    term3 = torch.trace(Gts @ Kst)  # scalar (trace over [n_t,n_s]@[n_s,n_t] -> [n_t,n_t])
    return term1 + term2 - 2.0 * term3


# =========================
# 4) 你的模型（改 forward: 同时跑 source/target，吐出多层特征）
# =========================
class CDP_Net(nn.Module):
    def __init__(self, num_classes=9):
        super(CDP_Net, self).__init__()
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
        self.classifier_1 = nn.Sequential(
            nn.Linear(256, 20),
            nn.SELU(),
        )
        self.classifier_2 = nn.Sequential(
            nn.Linear(20, 20),
            nn.SELU()
        )
        self.classifier_3 = nn.Sequential(
            nn.Linear(20, 20),
            nn.SELU()
        )
        self.final_classifier = nn.Sequential(
            nn.Linear(20, num_classes)
        )

    def forward_single(self, x):
        f0 = self.features(x)              # [B, 256]
        f1 = self.classifier_1(f0)         # [B, 20]
        f2 = self.classifier_2(f1)         # [B, 20]
        f3 = self.classifier_3(f2)         # [B, 20]
        logits = self.final_classifier(f3) # [B, C]
        return logits, (f0, f1, f2)

    def forward(self, source, target):
        s_logits, s_feats = self.forward_single(source)
        t_logits, t_feats = self.forward_single(target)
        return s_logits, t_logits, s_feats, t_feats


# =========================
# 5) Logit Adjustment (训练可用，验证禁用)
# =========================
def compute_adjustment_1(Y_profiling, tro, classes=9, eps=1e-12):
    counts = np.bincount(Y_profiling, minlength=classes).astype(np.float64)
    pi = counts / np.maximum(counts.sum(), 1.0)
    adjustments = np.log(np.power(pi, tro) + eps)
    return adjustments.astype(np.float32)


# =========================
# 6) Clock jitter
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
            print(str(trace_idx) + '/' + str(len(traces)))
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
# 7) DCAN-Finetune Train / Validation
# =========================
def build_finetune_cache(loader, num_iter_target, batch_size, trace_length, device):
    it = iter(loader)
    cache = torch.zeros((num_iter_target, batch_size, 1, trace_length), device=device)
    for i in range(num_iter_target):
        data_batch, _ = next(it)
        cache[i] = data_batch.to(device, non_blocking=True)
    return cache

def DCAN_train(epoch, model, optimizer,
               source_train_loader, target_finetune_loader,
               batch_size, trace_length,
               num_classes,
               lambda_cmmd=0.1, lambda_mi=0.2,
               gamma0=0.95, cmmd_reg=1e-3,
               adjust_flag=True, adjustments=None,
               log_interval=50, device="cuda"):

    model.train()
    clf_criterion = nn.CrossEntropyLoss()

    iter_source = iter(source_train_loader)
    iter_target = iter(target_finetune_loader)
    num_iter_target = len(target_finetune_loader)

    # 预取 target
    finetune_trace_all = torch.zeros((num_iter_target, batch_size, 1, trace_length), device=device)
    for i in range(num_iter_target):
        data_batch, _ = next(iter_target)
        finetune_trace_all[i] = data_batch.to(device, non_blocking=True)

    num_iter = len(source_train_loader)

    for i in range(1, num_iter + 1):
        source_data, source_label = next(iter_source)
        source_data = source_data.to(device, non_blocking=True)
        source_label = source_label.to(device, non_blocking=True)

        target_data = finetune_trace_all[(i - 1) % num_iter_target]

        optimizer.zero_grad(set_to_none=True)

        s_logits, t_logits, s_feats, t_feats = model(source_data, target_data)

        # ---- 1) Source CE
        s_logits_for_ce = s_logits
        if adjust_flag and (adjustments is not None):
            s_logits_for_ce = s_logits_for_ce + 1.0 * adjustments  # 仅训练使用（你原逻辑）

        clf_loss = clf_criterion(s_logits_for_ce, source_label)

        # ---- 2) Target MI (DCAN Eq.10)
        t_prob = torch.softmax(t_logits, dim=1)
        mi_loss = mutual_information_loss(t_prob)

        # ---- 3) CMMD with pseudo-label selection (gamma0)
        with torch.no_grad():
            conf, pseudo = torch.max(t_prob, dim=1)  # [B]
            sel_mask = conf > gamma0
            sel_idx = sel_mask.nonzero(as_tuple=False).squeeze(1)

        cmmd_total = s_logits.new_tensor(0.0)
        if sel_idx.numel() >= 2:
            # 取相同数量的 source 与 target 参与 CMMD（论文里两边 batch-size 一致）
            m = min(source_data.size(0), sel_idx.numel())
            if m >= 2:
                # 随机抽 m 个 source
                perm = torch.randperm(source_data.size(0), device=device)[:m]
                # 抽 m 个 target(高置信)
                sel_idx = sel_idx[:m]

                y_s_oh = one_hot(source_label[perm], num_classes=num_classes)
                y_t_oh = one_hot(pseudo[sel_idx], num_classes=num_classes)

                # 多层 CMMD（对应你原多层对齐的风格）
                for (zs, zt) in zip(s_feats, t_feats):
                    z_s = zs[perm]
                    z_t = zt[sel_idx]
                    cmmd_total = cmmd_total + cmmd_loss(
                        z_s, y_s_oh, z_t, y_t_oh,
                        reg=cmmd_reg
                    )

        # ---- Total loss
        loss = clf_loss + lambda_cmmd * cmmd_total + lambda_mi * mi_loss

        loss.backward()
        optimizer.step()

        if i % log_interval == 0:
            print(
                f"Train Epoch {epoch}: [{i * len(source_data)}/{len(source_train_loader) * batch_size} "
                f"({100. * i / len(source_train_loader):.0f}%)]\t"
                f"total_loss: {loss.item():.6f}\t"
                f"clf_loss: {clf_loss.item():.6f}\t"
                f"cmmd_loss: {cmmd_total.item():.6f}\t"
                f"mi_loss: {mi_loss.item():.6f}\t"
                f"sel_t: {int(sel_idx.numel())}"
            )

def DCAN_validation(model,
                    source_valid_loader, target_finetune_loader,
                    batch_size, trace_length,
                    num_classes,
                    lambda_cmmd=0.1, lambda_mi=0.2,
                    gamma0=0.95, cmmd_reg=1e-3,
                    device="cuda"):
    """
    验证阶段：按你的要求，不使用 logit adjustment
    """
    model.eval()
    clf_criterion = nn.CrossEntropyLoss()

    iter_source = iter(source_valid_loader)
    iter_target = iter(target_finetune_loader)
    num_iter_target = len(target_finetune_loader)

    finetune_trace_all = torch.zeros((num_iter_target, batch_size, 1, trace_length), device=device)
    for i in range(num_iter_target):
        data_batch, _ = next(iter_target)
        finetune_trace_all[i] = data_batch.to(device, non_blocking=True)

    num_iter = len(source_valid_loader)

    total_loss = 0.0
    total_clf = 0.0
    total_cmmd = 0.0
    total_mi = 0.0
    correct = 0

    with torch.no_grad():
        for i in range(1, num_iter + 1):
            source_data, source_label = next(iter_source)
            source_data = source_data.to(device, non_blocking=True)
            source_label = source_label.to(device, non_blocking=True)

            target_data = finetune_trace_all[(i - 1) % num_iter_target]

            s_logits, t_logits, s_feats, t_feats = model(source_data, target_data)

            # 1) CE (no logit adjustment)
            clf_loss = clf_criterion(s_logits, source_label)

            # 2) MI
            t_prob = torch.softmax(t_logits, dim=1)
            mi_loss = mutual_information_loss(t_prob)

            # 3) CMMD
            conf, pseudo = torch.max(t_prob, dim=1)
            sel_idx = (conf > gamma0).nonzero(as_tuple=False).squeeze(1)

            cmmd_total = s_logits.new_tensor(0.0)
            if sel_idx.numel() >= 2:
                m = min(source_data.size(0), sel_idx.numel())
                if m >= 2:
                    perm = torch.randperm(source_data.size(0), device=device)[:m]
                    sel_idx = sel_idx[:m]
                    y_s_oh = one_hot(source_label[perm], num_classes=num_classes)
                    y_t_oh = one_hot(pseudo[sel_idx], num_classes=num_classes)

                    for (zs, zt) in zip(s_feats, t_feats):
                        z_s = zs[perm]
                        z_t = zt[sel_idx]
                        cmmd_total = cmmd_total + cmmd_loss(
                            z_s, y_s_oh, z_t, y_t_oh,
                            reg=cmmd_reg
                        )

            loss = clf_loss + lambda_cmmd * cmmd_total + lambda_mi * mi_loss

            total_loss += loss.item()
            total_clf += clf_loss.item()
            total_cmmd += cmmd_total.item()
            total_mi += mi_loss.item()

            pred = s_logits.data.max(1)[1]
            correct += pred.eq(source_label.data.view_as(pred)).sum().item()

    total_loss /= len(source_valid_loader)
    total_clf /= len(source_valid_loader)
    total_cmmd /= len(source_valid_loader)
    total_mi /= len(source_valid_loader)

    print(
        f"Validation: total_loss: {total_loss:.4f}, "
        f"clf_loss: {total_clf:.4f}, "
        f"cmmd_loss: {total_cmmd:.4f}, "
        f"mi_loss: {total_mi:.4f}, "
        f"accuracy: {correct}/{len(source_valid_loader.dataset)} ({100. * correct / len(source_valid_loader.dataset):.2f}%)"
    )
    return total_loss


# =========================
# 8) Main
# =========================
if __name__ == '__main__':
    seed_everything(8)

    source_device_id = 0
    target_device_id = 1
    real_key_01 = 224
    real_key_02 = 224
    labeling_method = 'hw'

    # 你原本的超参
    lambda_ = 0.1          # 这里建议作为 lambda_cmmd 使用
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

    # DCAN 额外超参（按论文默认：λ0=0.1, λ1=0.2, γ0=0.95）
    lambda_cmmd = 0.01
    lambda_mi = 0.2
    gamma0 = 0.95
    cmmd_reg = 1e-3

    no_cuda = False
    cuda = (not no_cuda) and torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"

    # ---- load data
    X_train_source = np.load(source_file_path + 'X_train.npy')
    Y_train_source = np.load(source_file_path + 'Y_train.npy')
    X_attack_source = np.load(source_file_path + 'X_attack.npy')
    Y_attack_source = np.load(source_file_path + 'Y_attack.npy')

    # add clock jitter to target domain
    X_attack_target = addClockJitter(X_attack_source, clock_range, trace_length)
    Y_attack_target = Y_attack_source

    if labeling_method == 'identity':
        class_num = 256
    elif labeling_method == 'hw':
        class_num = 9
        Y_train_source = calculate_HW(Y_train_source)

    kwargs_source_train = {
        'trs_file': X_train_source[0:train_num, :],
        'label_file': Y_train_source[0:train_num],
        'trace_num': train_num,
        'trace_offset': trace_offset,
        'trace_length': trace_length,
    }
    kwargs_source_valid = {
        'trs_file': X_train_source[train_num:train_num + valid_num, :],
        'label_file': Y_train_source[train_num:train_num + valid_num],
        'trace_num': valid_num,
        'trace_offset': trace_offset,
        'trace_length': trace_length,
    }
    kwargs_target_finetune = {
        'trs_file': X_attack_target[0:target_finetune_num, :],
        'label_file': Y_attack_target[0:target_finetune_num],
        'trace_num': target_finetune_num,
        'trace_offset': trace_offset,
        'trace_length': trace_length,
    }

    source_train_loader = load_training(batch_size, kwargs_source_train)
    source_valid_loader = load_training(batch_size, kwargs_source_valid)
    target_finetune_loader = load_training(batch_size, kwargs_target_finetune)
    print('Load data complete!')

    # ---- logit adjustment (训练可用，验证不用)
    adjustments_np = compute_adjustment_1(Y_train_source[0:train_num], tro=1, classes=class_num)
    adjustments = torch.from_numpy(adjustments_np).view(1, -1).to(device)
    adjust_flag = True
    flag = "real" if adjust_flag else "fake"

    # ---- model & load pretrained
    model = CDP_Net(num_classes=class_num).to(device)

    pretrained_path = f'./models/{countermeasure}_{flag}_pre-trained_cpda_device{source_device_id}.pth'
    print("Loading model:", pretrained_path)
    checkpoint = None
    if os.path.exists(pretrained_path):
        checkpoint = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"Warning: Pretrained model not found at {pretrained_path}")

    optimizer = optim.Adam([
        {'params': model.features.parameters()},
        {'params': model.classifier_1.parameters()},
        {'params': model.classifier_2.parameters()},
        {'params': model.classifier_3.parameters()},
        {'params': model.final_classifier.parameters()}
    ], lr=lr)

    # 如果你的 checkpoint 里也存了 optimizer，且你希望接着训
    if checkpoint is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    min_loss = 1e9

    for epoch in range(1, finetune_epoch + 1):
        print(f'Train Epoch {epoch}:')
        DCAN_train(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            source_train_loader=source_train_loader,
            target_finetune_loader=target_finetune_loader,
            batch_size=batch_size,
            trace_length=trace_length,
            num_classes=class_num,
            lambda_cmmd=lambda_cmmd,
            lambda_mi=lambda_mi,
            gamma0=gamma0,
            cmmd_reg=cmmd_reg,
            adjust_flag=adjust_flag,
            adjustments=adjustments,
            log_interval=log_interval,
            device=device
        )

        with torch.no_grad():
            valid_loss = DCAN_validation(
                model=model,
                source_valid_loader=source_valid_loader,
                target_finetune_loader=target_finetune_loader,
                batch_size=batch_size,
                trace_length=trace_length,
                num_classes=class_num,
                lambda_cmmd=lambda_cmmd,
                lambda_mi=lambda_mi,
                gamma0=gamma0,
                cmmd_reg=cmmd_reg,
                device=device
            )

            if valid_loss < min_loss:
                min_loss = valid_loss
                if not os.path.exists('./models'):
                    os.makedirs('./models')
                save_path = f'./models/mutual{countermeasure}_{flag}_best_valid_loss_dcan_fine_tuned_cpda_device{source_device_id}_to_{target_device_id}.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, save_path)
                print("Saved best checkpoint to:", save_path)
