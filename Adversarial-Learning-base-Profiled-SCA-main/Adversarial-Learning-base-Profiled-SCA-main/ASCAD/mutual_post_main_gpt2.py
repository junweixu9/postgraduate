import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torch import nn, optim
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# =========================
# 0) Utils
# =========================
def seed_everything(seed: int = 8):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_adjustment_1(Y_profiling, tro, classes=9, eps=1e-12):
    counts = np.bincount(Y_profiling, minlength=classes).astype(np.float64)
    pi = counts / np.maximum(counts.sum(), 1.0)
    adjustments = np.log(np.power(pi, tro) + eps)  # log(pi^tau)
    return adjustments.astype(np.float32)


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
        trace_tensor = torch.from_numpy(trace).float().unsqueeze(0)
        label_tensor = torch.tensor(self.label_file[idx], dtype=torch.long)
        return trace_tensor, label_tensor

    def __len__(self):
        return self.trace_num


def load_training(batch_size, kwargs):
    data = TorchDataset(**kwargs)
    train_loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=True,
        drop_last=True, num_workers=4,
        pin_memory=True, persistent_workers=True
    )
    return train_loader


def load_testing(batch_size, kwargs):
    data = TorchDataset(**kwargs)
    test_loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=False,
        drop_last=True, num_workers=4,
        pin_memory=True, persistent_workers=True
    )
    return test_loader


# =========================
# 2) AES tables (用于 GE)
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
# 3) Clock Jitter
# =========================
def regulateMatrix(M, size):
    Z = np.zeros((len(M), size), dtype=np.float32)
    for enu, row in enumerate(M):
        row_len = len(row)
        if row_len <= size:
            Z[enu, :row_len] = row
        else:
            Z[enu, :] = row[:size]
    return Z

def addClockJitter(traces, clock_range, trace_length):
    print('Add clock jitters...')
    output_traces = []
    for trace_idx in range(len(traces)):
        if trace_idx % 2000 == 0:
            print(f"{trace_idx}/{len(traces)}")

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
# 4) DCAN: CMMD + Mutual Information
# =========================
def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(labels, num_classes=num_classes).float()

def linear_kernel(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return A @ B.t()

def rbf_kernel_multi_sigma(X: torch.Tensor, Y: torch.Tensor, sigmas=(0.1, 1.0, 10.0, 100.0, 1000.0)) -> torch.Tensor:
    x_norm = (X ** 2).sum(dim=1, keepdim=True)
    y_norm = (Y ** 2).sum(dim=1, keepdim=True)
    dist2 = x_norm + y_norm.t() - 2.0 * (X @ Y.t())
    dist2 = dist2.clamp_min(0.0)

    K = 0.0
    for s in sigmas:
        K = K + torch.exp(-dist2 / (2.0 * (s ** 2)))
    return K / float(len(sigmas))

def cmmd_loss(
    z_s: torch.Tensor, y_s_oh: torch.Tensor,
    z_t: torch.Tensor, y_t_oh: torch.Tensor,
    reg: float = 1e-3,
    sigmas=(0.1, 1.0, 10.0, 100.0, 1000.0),
) -> torch.Tensor:
    n_s = z_s.size(0)
    n_t = z_t.size(0)
    if n_s < 2 or n_t < 2:
        return z_s.new_tensor(0.0)

    Ks = rbf_kernel_multi_sigma(z_s, z_s, sigmas=sigmas)
    Kt = rbf_kernel_multi_sigma(z_t, z_t, sigmas=sigmas)
    Kst = rbf_kernel_multi_sigma(z_s, z_t, sigmas=sigmas)

    Ls = linear_kernel(y_s_oh, y_s_oh)
    Lt = linear_kernel(y_t_oh, y_t_oh)
    Lts = linear_kernel(y_t_oh, y_s_oh)  # [n_t, n_s]

    I_s = torch.eye(n_s, device=z_s.device, dtype=z_s.dtype)
    I_t = torch.eye(n_t, device=z_t.device, dtype=z_t.dtype)

    Ls_tilde = Ls + reg * I_s
    Lt_tilde = Lt + reg * I_t

    inv_Ls = torch.linalg.solve(Ls_tilde, I_s)
    inv_Lt = torch.linalg.solve(Lt_tilde, I_t)

    Gs = inv_Ls @ Ls @ inv_Ls
    Gt = inv_Lt @ Lt @ inv_Lt
    Gts = inv_Lt @ Lts @ inv_Ls  # [n_t, n_s]

    term1 = torch.trace(Gs @ Ks)
    term2 = torch.trace(Gt @ Kt)
    term3 = torch.trace((Gts @ Kst))
    return term1 + term2 - 2.0 * term3

def entropy_from_probs(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = p.clamp(min=eps, max=1.0)
    return -(p * p.log()).sum(dim=1)  # [B]

def mutual_information_loss(p_t: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # L_MI = E[H(p(y|x))] - H(E[p(y|x)])
    cond_ent = entropy_from_probs(p_t, eps=eps).mean()
    mean_p = p_t.mean(dim=0, keepdim=True)
    marg_ent = entropy_from_probs(mean_p, eps=eps).squeeze(0)
    return cond_ent - marg_ent


# =========================
# 5) Model: 保持你的 Net，并加一个 forward_pair 给 DCAN 用
# =========================
class Net(nn.Module):
    def __init__(self, num_classes=9):
        super(Net, self).__init__()
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

    def forward(self, x):
        f0 = self.features(x)             # [B,256]
        f1 = self.classifier_1(f0)        # [B,20]
        f2 = self.classifier_2(f1)        # [B,20]
        f3 = self.classifier_3(f2)        # [B,20]
        logits = self.final_classifier(f3)
        return logits

    def forward_single(self, x):
        f0 = self.features(x)
        f1 = self.classifier_1(f0)
        f2 = self.classifier_2(f1)
        f3 = self.classifier_3(f2)
        logits = self.final_classifier(f3)
        # 多层 CMMD（跟你原本多层对齐风格一致）
        return logits, (f0, f1, f2)

    def forward_pair(self, xs, xt):
        s_logits, s_feats = self.forward_single(xs)
        t_logits, t_feats = self.forward_single(xt)
        return s_logits, t_logits, s_feats, t_feats


# =========================
# 6) DCAN Fine-tuning Train/Validation
# =========================
def build_target_cache(target_loader, batch_size, trace_length, device):
    it = iter(target_loader)
    n = len(target_loader)
    cache = torch.zeros((n, batch_size, 1, trace_length), device=device)
    for i in range(n):
        x, _ = next(it)
        cache[i] = x.to(device, non_blocking=True)
    return cache

def dcan_train_epoch(
    epoch, model, optimizer,
    source_loader, target_finetune_loader,
    batch_size, trace_length, class_num,
    lambda_cmmd=0.1, lambda_mi=0.2, gamma0=0.95, cmmd_reg=1e-3,
    adjust_flag=True, adjustments=None,
    log_interval=50, device="cuda"
):
    model.train()
    ce = nn.CrossEntropyLoss()

    it_s = iter(source_loader)
    n_s = len(source_loader)

    # 预取 target finetune (你原本习惯)
    tgt_cache = build_target_cache(target_finetune_loader, batch_size, trace_length, device)
    n_t = tgt_cache.size(0)

    for i in range(1, n_s + 1):
        xs, ys = next(it_s)
        xs = xs.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True)
        xt = tgt_cache[(i - 1) % n_t]

        optimizer.zero_grad(set_to_none=True)

        s_logits, t_logits, s_feats, t_feats = model.forward_pair(xs, xt)

        # 1) 源域分类损失（训练期可选 logit adjustment）
        s_logits_for_ce = s_logits
        if adjust_flag and adjustments is not None:
            s_logits_for_ce = s_logits_for_ce + 1.0 * adjustments
        L_sc = ce(s_logits_for_ce, ys)

        # 2) 目标域互信息（让目标域预测更可分）
        t_prob = torch.softmax(t_logits, dim=1)
        L_mi = mutual_information_loss(t_prob)

        # 3) CMMD：只用高置信伪标签样本（> gamma0）
        with torch.no_grad():
            conf, pseudo = torch.max(t_prob, dim=1)
            sel = (conf > gamma0).nonzero(as_tuple=False).squeeze(1)

        L_cmmd = s_logits.new_tensor(0.0)
        if sel.numel() >= 2:
            m = min(xs.size(0), sel.numel())
            if m >= 2:
                perm = torch.randperm(xs.size(0), device=device)[:m]
                sel = sel[:m]
                y_s_oh = one_hot(ys[perm], class_num)
                y_t_oh = one_hot(pseudo[sel], class_num)

                for zs, zt in zip(s_feats, t_feats):
                    L_cmmd = L_cmmd + cmmd_loss(
                        zs[perm], y_s_oh,
                        zt[sel], y_t_oh,
                        reg=cmmd_reg
                    )

        loss = L_sc + lambda_cmmd * L_cmmd + lambda_mi * L_mi
        loss.backward()
        optimizer.step()

        if i % log_interval == 0:
            print(
                f"Train Epoch {epoch}: [{i*len(xs)}/{len(source_loader)*batch_size} ({100.*i/len(source_loader):.0f}%)]\t"
                f"total_loss: {loss.item():.6f}\t"
                f"sc_loss: {L_sc.item():.6f}\t"
                f"cmmd_loss: {L_cmmd.item():.6f}\t"
                f"mi_loss: {L_mi.item():.6f}\t"
                f"sel_t: {int(sel.numel())}"
            )

def dcan_validate(
    model, source_valid_loader, target_finetune_loader,
    batch_size, trace_length, class_num,
    lambda_cmmd=0.1, lambda_mi=0.2, gamma0=0.95, cmmd_reg=1e-3,
    log_interval=50, device="cuda"
):
    """
    验证期：按你的要求，不做 logit adjustment
    """
    model.eval()
    ce = nn.CrossEntropyLoss()

    it_s = iter(source_valid_loader)
    n_s = len(source_valid_loader)

    tgt_cache = build_target_cache(target_finetune_loader, batch_size, trace_length, device)
    n_t = tgt_cache.size(0)

    total_loss = 0.0
    total_sc = 0.0
    total_cmmd = 0.0
    total_mi = 0.0
    correct = 0

    with torch.no_grad():
        for i in range(1, n_s + 1):
            xs, ys = next(it_s)
            xs = xs.to(device, non_blocking=True)
            ys = ys.to(device, non_blocking=True)
            xt = tgt_cache[(i - 1) % n_t]

            s_logits, t_logits, s_feats, t_feats = model.forward_pair(xs, xt)

            L_sc = ce(s_logits, ys)

            t_prob = torch.softmax(t_logits, dim=1)
            L_mi = mutual_information_loss(t_prob)

            conf, pseudo = torch.max(t_prob, dim=1)
            sel = (conf > gamma0).nonzero(as_tuple=False).squeeze(1)

            L_cmmd = s_logits.new_tensor(0.0)
            if sel.numel() >= 2:
                m = min(xs.size(0), sel.numel())
                if m >= 2:
                    perm = torch.randperm(xs.size(0), device=device)[:m]
                    sel = sel[:m]
                    y_s_oh = one_hot(ys[perm], class_num)
                    y_t_oh = one_hot(pseudo[sel], class_num)
                    for zs, zt in zip(s_feats, t_feats):
                        L_cmmd = L_cmmd + cmmd_loss(zs[perm], y_s_oh, zt[sel], y_t_oh, reg=cmmd_reg)

            loss = L_sc + lambda_cmmd * L_cmmd + lambda_mi * L_mi

            total_loss += loss.item()
            total_sc += L_sc.item()
            total_cmmd += L_cmmd.item()
            total_mi += L_mi.item()

            pred = s_logits.data.max(1)[1]
            correct += pred.eq(ys.data.view_as(pred)).sum().item()

    total_loss /= len(source_valid_loader)
    total_sc /= len(source_valid_loader)
    total_cmmd /= len(source_valid_loader)
    total_mi /= len(source_valid_loader)

    print(
        f"Validation: total_loss: {total_loss:.4f}, sc_loss: {total_sc:.4f}, "
        f"cmmd_loss: {total_cmmd:.4f}, mi_loss: {total_mi:.4f}, "
        f"acc: {correct}/{len(source_valid_loader.dataset)} ({100.*correct/len(source_valid_loader.dataset):.2f}%)"
    )
    return total_loss


# =========================
# 7) Test / GE / Confusion (测试期不做 logit adjustment)
# =========================
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.ylim((len(classes) - 0.5, -0.5))
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predict label')
    plt.show()

def plot_guessing_entropy(preds, real_key, device_id, model_flag):
    num_averaged = 100
    trace_num_max = 5000

    if device_id == target_device_id:
        plaintext = plaintexts_target
    else:
        plaintext = plaintexts_source

    key_guesses = np.arange(256, dtype=np.int32)
    guessing_entropy = np.zeros((num_averaged, trace_num_max))
    success_flag = np.zeros((num_averaged, trace_num_max))

    for t in range(num_averaged):
        idx = np.random.choice(len(plaintext), trace_num_max, replace=False)
        selected_pt = plaintext[idx]
        selected_preds = preds[idx]

        state = selected_pt[:, np.newaxis] ^ key_guesses[np.newaxis, :]
        sbox_out = Sbox_np[state]

        if labeling_method == 'identity':
            labels = sbox_out
        else:
            labels = HW_byte_np[sbox_out]

        row_indices = np.arange(trace_num_max)[:, np.newaxis]
        probs = selected_preds[row_indices, labels]
        log_probs = np.log(probs + 1e-40)

        cumulative_scores = np.cumsum(log_probs, axis=0)
        real_key_scores = cumulative_scores[:, real_key][:, np.newaxis]
        ranks = np.sum(cumulative_scores > real_key_scores, axis=1)

        guessing_entropy[t, :] = ranks
        success_flag[t, :] = (ranks == 0).astype(int)

    avg_ge = np.mean(guessing_entropy, axis=0)
    avg_sr = np.mean(success_flag, axis=0)

    converge_idx = np.argmax(avg_ge < 1)
    if avg_ge[converge_idx] >= 1:
        print(f"[{model_flag}] GE did not converge to < 1 within {trace_num_max} traces.")
    else:
        print(f"[{model_flag}] Traces to reach GE < 1: {converge_idx + 1}")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(avg_ge, color='red')
    plt.xlabel('Number of trace')
    plt.ylabel('Guessing entropy')

    plt.subplot(1, 2, 2)
    plt.plot(avg_sr, color='red')
    plt.xlabel('Number of trace')
    plt.ylabel('Success rate')
    plt.show()


def test(model, device_id, disp_GE=True, model_flag='finetuned'):
    """
    测试期间：严格不使用 logit adjustment（你要求）
    """
    model.eval()
    clf_criterion = nn.CrossEntropyLoss()

    if device_id == source_device_id:
        test_num = source_test_num
        test_loader = source_test_loader
        real_key = real_key_01
        print("[TEST] on SOURCE")
    else:
        test_num = target_test_num
        test_loader = target_test_loader
        real_key = real_key_02
        print("[TEST] on TARGET")

    predlist = torch.zeros(0, dtype=torch.long, device='cpu')
    lbllist = torch.zeros(0, dtype=torch.long, device='cpu')
    test_preds_all = torch.zeros((test_num, class_num), dtype=torch.float, device='cpu')

    test_loss = 0.0
    correct = 0
    epoch = 0
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for data, label in test_loader:
            if cuda:
                data = data.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)

            logits = model(data)
            loss = clf_criterion(logits, label)
            test_loss += loss.item()

            pred = logits.data.max(1)[1]

            bs = data.size(0)
            start_idx = epoch * batch_size
            end_idx = start_idx + bs
            if end_idx <= test_preds_all.shape[0]:
                test_preds_all[start_idx:end_idx, :] = softmax(logits).cpu()

            predlist = torch.cat([predlist, pred.view(-1).cpu()])
            lbllist = torch.cat([lbllist, label.view(-1).cpu()])
            correct += pred.eq(label.data.view_as(pred)).cpu().sum().item()
            epoch += 1

    test_loss /= len(test_loader)
    print(f"Test loss: {test_loss:.4f}, accuracy: {correct}/{len(test_loader.dataset)} ({100.*correct/len(test_loader.dataset):.2f}%)\n")

    cm = confusion_matrix(lbllist.numpy(), predlist.numpy())
    plot_confusion_matrix(cm, classes=range(class_num))

    if disp_GE:
        plot_guessing_entropy(test_preds_all.numpy(), real_key, device_id, model_flag)


# =========================
# 8) Main
# =========================
if __name__ == '__main__':
    # ---------- 超参 ----------
    source_device_id = 0
    target_device_id = 1
    real_key_01 = 224
    real_key_02 = 224

    labeling_method = 'hw'       # 'hw' or 'identity'
    batch_size = 200
    lr = 0.001
    finetune_epoch = 30
    log_interval = 50

    train_num = 45000
    valid_num = 5000
    source_test_num = 10000
    target_finetune_num = 200
    target_test_num = 9000

    trace_offset = 0
    trace_length = 700

    countermeasure = '_clockjitter_level1'
    clock_range = 1
    source_file_path = './Data/ASCAD/'

    # ---------- DCAN 超参 ----------
    lambda_cmmd = 0.1
    lambda_mi = 0.2
    gamma0 = 0.95
    cmmd_reg = 1e-3

    # ---------- Logit adjustment：训练期可用；验证/测试禁用 ----------
    adjust_flag = True
    tro = 1.0

    # ---------- 环境 ----------
    no_cuda = False
    cuda = (not no_cuda) and torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    seed_everything(8)

    # ---------- load train/valid data ----------
    X_train_source = np.load(source_file_path + 'X_train.npy')
    Y_train_source = np.load(source_file_path + 'Y_train.npy')

    X_attack_source = np.load(source_file_path + 'X_attack.npy')
    Y_attack_source = np.load(source_file_path + 'Y_attack.npy')

    if labeling_method == 'identity':
        class_num = 256
    else:
        class_num = 9
        Y_train_source = calculate_HW(Y_train_source)
        Y_attack_source = calculate_HW(Y_attack_source)

    # target domain = clock jittered attack traces
    X_attack_target = addClockJitter(X_attack_source, clock_range, trace_length)
    Y_attack_target = Y_attack_source

    # ---------- plaintexts for GE ----------
    plaintexts_source = np.load(source_file_path + 'plaintexts_attack.npy')[:, 2]
    # target test plaintext slice 要对齐 target_test_loader 的切片
    plaintexts_target = np.load(source_file_path + 'plaintexts_attack.npy')[target_finetune_num: target_finetune_num + target_test_num, 2]

    # ---------- dataloaders: train/valid ----------
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

    # ---------- dataloaders: test ----------
    kwargs_source_test = {
        'trs_file': X_attack_source,
        'label_file': Y_attack_source,
        'trace_num': source_test_num,
        'trace_offset': trace_offset,
        'trace_length': trace_length,
    }
    kwargs_target_test = {
        'trs_file': X_attack_target[target_finetune_num: target_finetune_num + target_test_num, :],
        'label_file': Y_attack_target[target_finetune_num: target_finetune_num + target_test_num],
        'trace_num': target_test_num,
        'trace_offset': trace_offset,
        'trace_length': trace_length,
    }
    source_test_loader = load_testing(batch_size, kwargs_source_test)
    target_test_loader = load_testing(batch_size, kwargs_target_test)
    print('Load data complete!')

    # ---------- logit adjustment tensor（训练期用） ----------
    adjustments = None
    if adjust_flag:
        adj_np = compute_adjustment_1(Y_train_source[0:train_num], tro=tro, classes=class_num)
        adjustments = torch.from_numpy(adj_np).view(1, -1).to(device)

    flag = "real" if adjust_flag else "fake"

    # ---------- model ----------
    model = Net(num_classes=class_num).to(device)
    print('Construct model complete')

    # ---------- （可选）加载你原来预训练 checkpoint ----------
    pretrained_path = f'./models/{countermeasure}_{flag}_pre-trained_cpda_device{source_device_id}.pth'
    checkpoint = None
    if os.path.exists(pretrained_path):
        print("Loading pretrained:", pretrained_path)
        checkpoint = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Warning: no pretrained checkpoint, start from scratch.")

    optimizer = optim.Adam([
        {'params': model.features.parameters()},
        {'params': model.classifier_1.parameters()},
        {'params': model.classifier_2.parameters()},
        {'params': model.classifier_3.parameters()},
        {'params': model.final_classifier.parameters()}
    ], lr=lr)

    if checkpoint is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # ---------- DCAN fine-tuning ----------
    min_loss = 1e9
    best_path = f'./models/mutual{countermeasure}_{flag}_best_valid_loss_dcan_fine_tuned_cpda_device{source_device_id}_to_{target_device_id}.pth'
    os.makedirs('./models', exist_ok=True)

    # ---------- load best & test ----------
    print("\n=== Load best DCAN and test (NO logit adjustment in test) ===")
    ck = torch.load(best_path, map_location=device)
    model.load_state_dict(ck['model_state_dict'])

    with torch.no_grad():
        print('Result on source device:')
        test(model, source_device_id, model_flag='dcan_finetuned_source')
        print('Result on target device:')
        test(model, target_device_id, model_flag='dcan_finetuned_target')
