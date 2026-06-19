import os
from torch.utils.data import Dataset
import torch
from torch import optim
import numpy as np
from torch import nn
import torch.nn.functional as F
import random

# 【优化】开启 cudnn 自动寻优
torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# ==============================
# 0) Dataset / DataLoader
# ==============================
class TorchDataset(Dataset):
    def __init__(self, trs_file, label_file, trace_num, trace_offset, trace_length):
        self.trs_file = trs_file
        self.label_file = label_file
        self.trace_num = trace_num
        self.trace_offset = trace_offset
        self.trace_length = trace_length

    def __getitem__(self, i):
        index = i % self.trace_num
        trace = self.trs_file[index, self.trace_offset: self.trace_offset + self.trace_length]
        trace_tensor = torch.from_numpy(trace).float().unsqueeze(0)  # (1, L)
        label_tensor = torch.tensor(self.label_file[index], dtype=torch.long)
        return trace_tensor, label_tensor

    def __len__(self):
        return self.trace_num


def load_training(batch_size, kwargs):
    data = TorchDataset(**kwargs)
    train_loader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    return train_loader


def load_testing(batch_size, kwargs):
    data = TorchDataset(**kwargs)
    test_loader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    return test_loader


# ==============================
# 1) HW lookup (vectorized)
# ==============================
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


def compute_adjustment_1(Y_profiling, tro, classes=9, eps=1e-12):
    counts = np.bincount(Y_profiling, minlength=classes).astype(np.float64)
    pi = counts / np.maximum(counts.sum(), 1.0)
    adjustments = np.log(np.power(pi, tro) + eps)  # log(pi^tau)
    return adjustments.astype(np.float32)


# ==============================
# 2) DCAN-style: CMMD + Mutual Information
# ==============================
def rbf_mixture_kernel(x: torch.Tensor, y: torch.Tensor, sigmas=(0.1, 1.0, 10.0, 100.0, 1000.0)):
    """
    x: (n,d), y: (m,d) -> K: (n,m)
    """
    x_norm = (x ** 2).sum(dim=1, keepdim=True)      # (n,1)
    y_norm = (y ** 2).sum(dim=1, keepdim=True).T    # (1,m)
    dist2 = x_norm + y_norm - 2.0 * (x @ y.T)       # (n,m)
    k = 0.0
    for s in sigmas:
        k = k + torch.exp(-dist2 / (2.0 * (s ** 2)))
    return k / float(len(sigmas))


def label_linear_kernel(Ya: torch.Tensor, Yb: torch.Tensor):
    return Ya @ Yb.T


def one_hot(labels: torch.Tensor, num_classes: int):
    return F.one_hot(labels.long(), num_classes=num_classes).float()


def cmmd_loss(zs: torch.Tensor,
              ys: torch.Tensor,
              zt: torch.Tensor,
              yt: torch.Tensor,
              lam: float = 1e-3,
              sigmas=(0.1, 1.0, 10.0, 100.0, 1000.0)):
    """
    Batch CMMD estimator (trace form). Requires equal batch size n for zs and zt.
    """
    n = zs.size(0)
    device = zs.device
    I = torch.eye(n, device=device)

    Ks = rbf_mixture_kernel(zs, zs, sigmas=sigmas)
    Kt = rbf_mixture_kernel(zt, zt, sigmas=sigmas)
    Kst = rbf_mixture_kernel(zs, zt, sigmas=sigmas)

    Ls = label_linear_kernel(ys, ys)
    Lt = label_linear_kernel(yt, yt)
    Lts = label_linear_kernel(yt, ys)  # (t,s)

    invLs = torch.linalg.inv(Ls + lam * I)
    invLt = torch.linalg.inv(Lt + lam * I)

    Gs = invLs @ Ls @ invLs
    Gt = invLt @ Lt @ invLt
    Gts = invLt @ Lts @ invLs

    return torch.trace(Gs @ Ks) + torch.trace(Gt @ Kt) - 2.0 * torch.trace(Gts @ Kst)


def mutual_information_loss(p: torch.Tensor, eps: float = 1e-12):
    """
    L_MI = mean_i H(p_i) - H(mean_p). Minimize this <=> maximize MI
    """
    p = torch.clamp(p, eps, 1.0)
    H_cond = -(p * torch.log(p)).sum(dim=1).mean()
    p_bar = torch.clamp(p.mean(dim=0), eps, 1.0)
    H_marg = -(p_bar * torch.log(p_bar)).sum()
    return H_cond - H_marg


# ==============================
# 3) Model (final_classifier uses Sequential to match old checkpoint keys)
# ==============================
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
        self.classifier_1 = nn.Sequential(nn.Linear(256, 20), nn.SELU())
        self.classifier_2 = nn.Sequential(nn.Linear(20, 20), nn.SELU())
        self.classifier_3 = nn.Sequential(nn.Linear(20, 20), nn.SELU())

        # 【关键】用 Sequential 包一层 Linear，匹配 checkpoint: final_classifier.0.weight/bias
        self.final_classifier = nn.Sequential(
            nn.Linear(20, num_classes)
        )

    def forward(self, source, target):
        # source
        zs0 = self.features(source)      # (bs,256)
        zs1 = self.classifier_1(zs0)     # (bs,20)
        zs2 = self.classifier_2(zs1)     # (bs,20)
        zs3 = self.classifier_3(zs2)     # (bs,20)
        source_logits = self.final_classifier(zs3)

        # target
        zt0 = self.features(target)
        zt1 = self.classifier_1(zt0)
        zt2 = self.classifier_2(zt1)
        zt3 = self.classifier_3(zt2)
        target_logits = self.final_classifier(zt3)

        feats_s = [zs0, zs1, zs2]  # 保持你原来对齐 3 层
        feats_t = [zt0, zt1, zt2]
        return source_logits, target_logits, feats_s, feats_t


# ==============================
# 4) Train / Validation
#    loss = CE(source) + lambda0 * CMMD + lambda1 * MI
# ==============================
def CDP_train(epoch, model,
              lambda0=0.1, lambda1=0.2, gamma0=0.95,
              cmmd_lam=1e-3, sigmas=(0.1, 1.0, 10.0, 100.0, 1000.0)):
    model.train()  # 【修正】必须 train()

    iter_source = iter(source_train_loader)
    iter_target = iter(target_finetune_loader)
    num_iter_target = len(target_finetune_loader)

    # prefetch target
    finetune_trace_all = torch.zeros((num_iter_target, batch_size, 1, trace_length))
    for j in range(num_iter_target):
        data_batch, _ = next(iter_target)
        finetune_trace_all[j] = data_batch

    num_iter = len(source_train_loader)
    clf_criterion = nn.CrossEntropyLoss()

    for it in range(1, num_iter + 1):
        source_data, source_label = next(iter_source)
        target_data = finetune_trace_all[(it - 1) % num_iter_target]

        if cuda:
            source_data = source_data.cuda(non_blocking=True)
            source_label = source_label.cuda(non_blocking=True)
            target_data = target_data.cuda(non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        source_logits, target_logits, feats_s, feats_t = model(source_data, target_data)

        # CE on source
        if adjust_flag:
            source_logits = source_logits + adjustments  # adjustments: (C,)
        clf_loss = clf_criterion(source_logits, source_label)

        # MI on target
        p_t = F.softmax(target_logits, dim=1)
        mi_loss = mutual_information_loss(p_t)

        # CMMD with pseudo labels (high confidence)
        conf, yhat = torch.max(p_t.detach(), dim=1)
        mask = conf > gamma0

        cmmd_total = torch.tensor(0.0, device=source_logits.device)
        if mask.sum().item() > 1:
            idx_t_all = torch.where(mask)[0]
            n_sel = min(idx_t_all.numel(), source_label.size(0))

            perm_t = torch.randperm(idx_t_all.numel(), device=idx_t_all.device)[:n_sel]
            idx_t = idx_t_all[perm_t]
            idx_s = torch.randperm(source_label.size(0), device=source_label.device)[:n_sel]

            ys_oh = one_hot(source_label[idx_s], num_classes=class_num)
            yt_oh = one_hot(yhat[idx_t], num_classes=class_num)

            for k in range(len(feats_s)):
                zs = feats_s[k][idx_s]
                zt = feats_t[k][idx_t]
                cmmd_total = cmmd_total + cmmd_loss(zs, ys_oh, zt, yt_oh, lam=cmmd_lam, sigmas=sigmas)

        loss = clf_loss + lambda0 * cmmd_total + lambda1 * mi_loss
        loss.backward()
        optimizer.step()

        if it % log_interval == 0:
            print(
                f"Train Epoch {epoch}: [{it}/{num_iter}] "
                f"total={loss.item():.6f} "
                f"clf={clf_loss.item():.6f} "
                f"cmmd={cmmd_total.item():.6f} "
                f"mi={mi_loss.item():.6f} "
                f"pseudo_used={int(mask.sum().item())}/{mask.numel()}"
            )


def CDP_validation(model,
                   lambda0=0.1, lambda1=0.2, gamma0=0.95,
                   cmmd_lam=1e-3, sigmas=(0.1, 1.0, 10.0, 100.0, 1000.0)):
    clf_criterion = nn.CrossEntropyLoss()
    model.eval()

    iter_source = iter(source_valid_loader)
    iter_target = iter(target_finetune_loader)
    num_iter_target = len(target_finetune_loader)

    finetune_trace_all = torch.zeros((num_iter_target, batch_size, 1, trace_length))
    for j in range(num_iter_target):
        data_batch, _ = next(iter_target)
        finetune_trace_all[j] = data_batch

    num_iter = len(source_valid_loader)
    total_loss = 0.0
    total_clf = 0.0
    total_cmmd = 0.0
    total_mi = 0.0
    correct = 0

    with torch.no_grad():
        for it in range(1, num_iter + 1):
            source_data, source_label = next(iter_source)
            target_data = finetune_trace_all[(it - 1) % num_iter_target]

            if cuda:
                source_data = source_data.cuda(non_blocking=True)
                source_label = source_label.cuda(non_blocking=True)
                target_data = target_data.cuda(non_blocking=True)

            source_logits, target_logits, feats_s, feats_t = model(source_data, target_data)

            if adjust_flag:
                source_logits = source_logits + adjustments

            clf_loss = clf_criterion(source_logits, source_label)

            p_t = F.softmax(target_logits, dim=1)
            mi_loss = mutual_information_loss(p_t)

            conf, yhat = torch.max(p_t, dim=1)
            mask = conf > gamma0

            cmmd_total = torch.tensor(0.0, device=source_logits.device)
            if mask.sum().item() > 1:
                idx_t_all = torch.where(mask)[0]
                n_sel = min(idx_t_all.numel(), source_label.size(0))
                idx_t = idx_t_all[:n_sel]
                idx_s = torch.arange(n_sel, device=source_label.device)

                ys_oh = one_hot(source_label[idx_s], num_classes=class_num)
                yt_oh = one_hot(yhat[idx_t], num_classes=class_num)

                for k in range(len(feats_s)):
                    zs = feats_s[k][idx_s]
                    zt = feats_t[k][idx_t]
                    cmmd_total = cmmd_total + cmmd_loss(zs, ys_oh, zt, yt_oh, lam=cmmd_lam, sigmas=sigmas)

            loss = clf_loss + lambda0 * cmmd_total + lambda1 * mi_loss

            total_loss += loss.item()
            total_clf += clf_loss.item()
            total_cmmd += cmmd_total.item()
            total_mi += mi_loss.item()

            pred = source_logits.argmax(dim=1)
            correct += pred.eq(source_label).sum().item()

    n_batches = len(source_valid_loader)
    total_loss /= n_batches
    total_clf /= n_batches
    total_cmmd /= n_batches
    total_mi /= n_batches

    acc = 100.0 * correct / len(source_valid_loader.dataset)
    print(
        f"Validation: total={total_loss:.4f}, clf={total_clf:.4f}, "
        f"cmmd={total_cmmd:.4f}, mi={total_mi:.4f}, acc={acc:.2f}%"
    )
    return total_loss


# ==============================
# 5) Clock jitter (unchanged)
# ==============================
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


def regulateMatrix(M, size):
    maxlen = size
    Z = np.zeros((len(M), maxlen))
    for enu, row in enumerate(M):
        if len(row) <= maxlen:
            Z[enu, :len(row)] += row
        else:
            Z[enu, :] += row[:maxlen]
    return Z


# ==============================
# 6) Main
# ==============================
if __name__ == '__main__':
    # ---- config ----
    source_device_id = 0
    target_device_id = 1

    labeling_method = 'hw'

    # DCAN-like hyperparams
    lambda0 = 0.1
    lambda1 = 0.2
    gamma0 = 0.95
    cmmd_lam = 1e-3
    sigmas = (0.1, 1.0, 10.0, 100.0, 1000.0)

    batch_size = 200
    finetune_epoch = 30
    lr = 0.001
    log_interval = 50

    train_num = 45000
    valid_num = 5000
    target_finetune_num = 200

    trace_offset = 0
    trace_length = 700

    countermeasure = '_clockjitter_level1'
    clock_range = 1
    source_file_path = './Data/ASCAD/'

    # logit adjustment (optional)
    adjust_flag = True
    tro = 1.0
    adjustments = None

    no_cuda = False
    cuda = (not no_cuda) and torch.cuda.is_available()
    seed = 8
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    # ---- load data ----
    X_train_source = np.load(source_file_path + 'X_train.npy')
    Y_train_source = np.load(source_file_path + 'Y_train.npy')
    X_attack_source = np.load(source_file_path + 'X_attack.npy')
    Y_attack_source = np.load(source_file_path + 'Y_attack.npy')

    X_attack_target = addClockJitter(X_attack_source, clock_range, trace_length)

    if labeling_method == 'identity':
        class_num = 256
    elif labeling_method == 'hw':
        class_num = 9
        Y_train_source = calculate_HW(Y_train_source)
    else:
        raise ValueError("labeling_method must be 'identity' or 'hw'")

    if adjust_flag:
        adj_np = compute_adjustment_1(Y_train_source[:train_num], tro, classes=class_num)
        adjustments = torch.from_numpy(adj_np)
        if cuda:
            adjustments = adjustments.cuda(non_blocking=True)

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
    # target labels are NOT used (UDA). Put zeros to avoid accidental leakage.
    kwargs_target_finetune = {
        'trs_file': X_attack_target[0:target_finetune_num, :],
        'label_file': np.zeros((target_finetune_num,), dtype=np.int64),
        'trace_num': target_finetune_num,
        'trace_offset': trace_offset,
        'trace_length': trace_length,
    }

    source_train_loader = load_training(batch_size, kwargs_source_train)
    source_valid_loader = load_training(batch_size, kwargs_source_valid)
    target_finetune_loader = load_training(batch_size, kwargs_target_finetune)
    print('Load data complete!')

    # ---- model ----
    model = CDP_Net(num_classes=class_num)

    pretrained_path = ('./models/' + str(adjust_flag) + '_' + str(countermeasure) +
                       '_pre-trained_cpda_device{}.pth'.format(source_device_id))
    print("Loading model:", pretrained_path)

    checkpoint = None
    if os.path.exists(pretrained_path):
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Pretrained model loaded.")
    else:
        print(f"Warning: Pretrained model not found at {pretrained_path}. Training from scratch.")

    if cuda:
        model.cuda()

    # ---- optimizer ----
    # 【关键修复】fine-tune 只加载模型权重，optimizer 重新初始化，不加载 optimizer_state_dict
    optimizer = optim.Adam([
        {'params': model.features.parameters()},
        {'params': model.classifier_1.parameters()},
        {'params': model.classifier_2.parameters()},
        {'params': model.classifier_3.parameters()},
        {'params': model.final_classifier.parameters()},
    ], lr=lr)
    print("Fine-tune mode: optimizer state NOT loaded (avoids param_groups mismatch).")

    # ---- train loop ----
    min_loss = 1e18
    if not os.path.exists('./models'):
        os.makedirs('./models')

    for epoch in range(1, finetune_epoch + 1):
        print(f'\nTrain Epoch {epoch}:')
        CDP_train(epoch, model,
                  lambda0=lambda0, lambda1=lambda1, gamma0=gamma0,
                  cmmd_lam=cmmd_lam, sigmas=sigmas)

        with torch.no_grad():
            valid_loss = CDP_validation(model,
                                        lambda0=lambda0, lambda1=lambda1, gamma0=gamma0,
                                        cmmd_lam=cmmd_lam, sigmas=sigmas)

        if valid_loss < min_loss:
            min_loss = valid_loss
            save_path = ('./models/' + str(adjust_flag) + '_' + str(countermeasure) +
                         '_best_valid_loss_fine_tuned_cpda_gpt_device{}_to_{}.pth'.format(source_device_id, target_device_id))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, save_path)
            print(f"[Saved] epoch={epoch}, valid_loss={valid_loss:.6f} -> {save_path}")
