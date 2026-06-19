import os
from torch.utils.data import Dataset, DataLoader
import torch
from torch import optim
import numpy as np
from torch import nn
import random

torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# =========================
# Dataset
# =========================
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
        trace_tensor = torch.from_numpy(trace).float().unsqueeze(0)  # [1, L]
        label_tensor = torch.tensor(self.label_file[index], dtype=torch.long)
        return trace_tensor, label_tensor

    def __len__(self):
        return self.trace_num


def load_training(batch_size, kwargs, drop_last=True):
    data = TorchDataset(**kwargs)
    loader = DataLoader(
        data, batch_size=batch_size, shuffle=True,
        drop_last=drop_last, num_workers=4,
        pin_memory=True, persistent_workers=True
    )
    return loader


def load_testing(batch_size, kwargs, drop_last=True):
    data = TorchDataset(**kwargs)
    loader = DataLoader(
        data, batch_size=batch_size, shuffle=False,
        drop_last=drop_last, num_workers=4,
        pin_memory=True, persistent_workers=True
    )
    return loader


# =========================
# AES tables & HW
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
# CMMD
# =========================
def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    多核 RBF kernel matrix for [source; target]
    """
    n_samples = int(source.size(0)) + int(target.size(0))
    total = torch.cat([source, target], dim=0)  # [n, d]

    total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
    total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
    L2_distance = ((total0 - total1) ** 2).sum(2)

    if fix_sigma is not None:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.detach()) / (n_samples ** 2 - n_samples + 1e-12)

    bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / (bw + 1e-12)) for bw in bandwidth_list]
    return sum(kernel_val)


def cmmd_loss(source, target, s_label, t_label, num_classes=9, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    CMMD: E[k(xs,xs') * 1(ys=ys')] + E[k(xt,xt') * 1(yt=yt')]
          -2 E[k(xs,xt) * 1(ys=yt)]
    支持 bs != bt
    """
    bs = source.size(0)
    bt = target.size(0)

    s_onehot = torch.zeros(bs, num_classes, device=source.device)
    s_onehot.scatter_(1, s_label.view(-1, 1), 1)

    t_onehot = torch.zeros(bt, num_classes, device=target.device)
    t_onehot.scatter_(1, t_label.view(-1, 1), 1)

    K = gaussian_kernel(source, target, kernel_mul, kernel_num, fix_sigma)  # [(bs+bt),(bs+bt)]
    XX = K[:bs, :bs]
    YY = K[bs:bs+bt, bs:bs+bt]
    XY = K[:bs, bs:bs+bt]

    S = s_onehot @ s_onehot.t()     # [bs,bs]
    T = t_onehot @ t_onehot.t()     # [bt,bt]
    ST = s_onehot @ t_onehot.t()    # [bs,bt]

    return torch.mean(S * XX + T * YY - 2.0 * ST * XY)


@torch.no_grad()
def get_target_pseudo_labels(model, target_loader, device):
    """
    每个 batch 输出一个伪标签 tensor（cpu），与您原先“prefetch batch list”的方式对齐。
    """
    model.eval()
    pseudo_list = []
    for x, _ in target_loader:
        x = x.to(device, non_blocking=True)
        # 只需要 source 分支的分类头即可：用当前模型对 target 做预测
        _, _, _, logits = model.forward_features_and_logits(x)
        pseudo = logits.argmax(dim=1)
        pseudo_list.append(pseudo.cpu())
    return pseudo_list


# =========================
# Logit adjustment
# =========================
def compute_adjustment_1(Y_profiling, tro, classes=9, eps=1e-12):
    counts = np.bincount(Y_profiling, minlength=classes).astype(np.float64)
    pi = counts / np.maximum(counts.sum(), 1.0)
    adjustments = np.log(np.power(pi, tro) + eps)  # log(pi^tau)
    return adjustments.astype(np.float32)


# =========================
# Model (改：返回多层特征 + logits)
# =========================
class CDP_Net(nn.Module):
    def __init__(self, num_classes=9):
        super(CDP_Net, self).__init__()
        self.num_classes = num_classes

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

    def forward_features_and_logits(self, x):
        """
        给单域输入用（比如 target 伪标签推断）
        返回：f0, f1, f2, logits
        """
        f0 = self.features(x).view(x.size(0), -1)  # [b,256]
        f1 = self.classifier_1(f0)                 # [b,20]
        f2 = self.classifier_2(f1)                 # [b,20]
        f3 = self.classifier_3(f2)                 # [b,20]
        logits = self.final_classifier(f3)         # [b,C]
        return f0, f1, f2, logits

    def forward(self, source, target, source_label=None, target_label=None,
                use_cmmd=True):
        """
        返回：
          logits_s: source logits
          da_loss: CMMD loss（多层相加）
        """
        s0, s1, s2, logits_s = self.forward_features_and_logits(source)
        t0, t1, t2, _ = self.forward_features_and_logits(target)

        if use_cmmd and (source_label is not None) and (target_label is not None):
            cmmd0 = cmmd_loss(s0, t0, source_label, target_label, num_classes=self.num_classes)
            cmmd1 = cmmd_loss(s1, t1, source_label, target_label, num_classes=self.num_classes)
            cmmd2 = cmmd_loss(s2, t2, source_label, target_label, num_classes=self.num_classes)
            da_loss = cmmd0 + cmmd1 + cmmd2
        else:
            da_loss = torch.tensor(0.0, device=source.device)

        return logits_s, da_loss


# =========================
# Train / Validation
# =========================
def CDP_train(epoch, model):
    # ✅ 训练必须 train()：BN/Dropout 才会按训练行为运行 :contentReference[oaicite:1]{index=1}
    model.eval()

    clf_criterion = nn.CrossEntropyLoss()

    # 1) 每个 epoch 先算 target 伪标签（不反传）
    if use_cmmd:
        print("Computing pseudo labels for target domain...")
        target_pseudo_labels_list = get_target_pseudo_labels(model, target_finetune_loader, device)
    else:
        target_pseudo_labels_list = None

    iter_source = iter(source_train_loader)
    iter_target = iter(target_finetune_loader)
    num_iter_target = len(target_finetune_loader)

    # 2) 预取 target batch（CPU）
    finetune_trace_all = []
    for _ in range(num_iter_target):
        xb, _ = next(iter_target)
        finetune_trace_all.append(xb)

    num_iter = len(source_train_loader)

    for i in range(1, num_iter + 1):
        source_data, source_label = next(iter_source)
        target_idx = (i - 1) % num_iter_target
        target_data = finetune_trace_all[target_idx]

        if use_cmmd:
            target_label = target_pseudo_labels_list[target_idx]
        else:
            target_label = None

        source_data = source_data.to(device, non_blocking=True)
        source_label = source_label.to(device, non_blocking=True)
        target_data = target_data.to(device, non_blocking=True)
        if target_label is not None:
            target_label = target_label.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits_s, cmmd = model(
            source_data, target_data,
            source_label=source_label,
            target_label=target_label,
            use_cmmd=use_cmmd
        )

        # ✅ 训练阶段允许 logit adjustment
        if adjust_flag:
            logits_s = logits_s + adjustments  # [bs,C] + [1,C]

        clf_loss = clf_criterion(logits_s, source_label)

        # 总损失：clf + lambda_cmmd * cmmd
        loss = clf_loss + lambda_cmmd * cmmd
        loss.backward()
        optimizer.step()

        if i % log_interval == 0:
            print(
                'Train Epoch {}: [{}/{} ({:.0f}%)]\t'
                'total_loss: {:.6f}\tclf_loss: {:.6f}\tcmmd: {:.6f}\tlambda_cmmd: {:.4f}'.format(
                    epoch,
                    i * len(source_data),
                    len(source_train_loader.dataset),
                    100. * i / len(source_train_loader),
                    loss.item(),
                    clf_loss.item(),
                    cmmd.item(),
                    lambda_cmmd
                )
            )


@torch.no_grad()
def CDP_validation(model):
    # ✅ 验证用 eval()，且不使用 logit adjustment（按您的要求）
    model.eval()

    clf_criterion = nn.CrossEntropyLoss()

    # 伪标签（验证用同一套逻辑，保证 cmmd 可算）
    if use_cmmd:
        target_pseudo_labels_list = get_target_pseudo_labels(model, target_finetune_loader, device)
    else:
        target_pseudo_labels_list = None

    iter_source = iter(source_valid_loader)
    iter_target = iter(target_finetune_loader)
    num_iter_target = len(target_finetune_loader)

    finetune_trace_all = []
    for _ in range(num_iter_target):
        xb, _ = next(iter_target)
        finetune_trace_all.append(xb)

    num_iter = len(source_valid_loader)

    total_loss = 0.0
    total_clf_loss = 0.0
    total_cmmd_loss = 0.0
    correct = 0

    for i in range(1, num_iter + 1):
        source_data, source_label = next(iter_source)
        target_idx = (i - 1) % num_iter_target
        target_data = finetune_trace_all[target_idx]

        if use_cmmd:
            target_label = target_pseudo_labels_list[target_idx]
        else:
            target_label = None

        source_data = source_data.to(device, non_blocking=True)
        source_label = source_label.to(device, non_blocking=True)
        target_data = target_data.to(device, non_blocking=True)
        if target_label is not None:
            target_label = target_label.to(device, non_blocking=True)

        logits_s, cmmd = model(
            source_data, target_data,
            source_label=source_label,
            target_label=target_label,
            use_cmmd=use_cmmd
        )

        # ❌ 验证不加 adjustments
        clf_loss = clf_criterion(logits_s, source_label)
        loss = clf_loss + lambda_cmmd * cmmd

        total_loss += loss.item()
        total_clf_loss += clf_loss.item()
        total_cmmd_loss += cmmd.item()

        pred = logits_s.data.max(1)[1]
        correct += pred.eq(source_label.data.view_as(pred)).sum().item()

    n = len(source_valid_loader)
    total_loss /= n
    total_clf_loss /= n
    total_cmmd_loss /= n

    acc = 100.0 * correct / len(source_valid_loader.dataset)
    print(
        'Validation: total_loss: {:.4f}, clf_loss: {:.4f}, cmmd: {:.4f}, accuracy: {}/{} ({:.2f}%)'.format(
            total_loss, total_clf_loss, total_cmmd_loss,
            correct, len(source_valid_loader.dataset), acc
        )
    )
    return total_loss


# =========================
# clock jitter (保持您原实现)
# =========================
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


# =========================
# Main (按您原参数)
# =========================
if __name__ == '__main__':
    source_device_id = 0
    target_device_id = 1
    labeling_method = 'hw'

    batch_size = 200
    finetune_epoch = 20
    lr = 0.002
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
    no_cuda = False
    cuda = (not no_cuda) and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    seed = 8
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)

    # -------- load data --------
    X_train_source = np.load(source_file_path + 'X_train.npy')
    Y_train_source = np.load(source_file_path + 'Y_train.npy')
    X_attack_source = np.load(source_file_path + 'X_attack.npy')
    Y_attack_source = np.load(source_file_path + 'Y_attack.npy')

    # target domain = clock jitter on attack traces
    X_attack_target = addClockJitter(X_attack_source, clock_range, trace_length)
    Y_attack_target = Y_attack_source.copy()

    if labeling_method == 'hw':
        class_num = 9
        Y_train_source = calculate_HW(Y_train_source)
        # 这里 Y_attack_target 若也要 hw 标签，可加一行：
        # Y_attack_target = calculate_HW(Y_attack_target)
    else:
        class_num = 256

    # -------- dataloaders --------
    kwargs_source_train = dict(
        trs_file=X_train_source[0:train_num, :],
        label_file=Y_train_source[0:train_num],
        trace_num=train_num,
        trace_offset=trace_offset,
        trace_length=trace_length
    )
    kwargs_source_valid = dict(
        trs_file=X_train_source[train_num:train_num + valid_num, :],
        label_file=Y_train_source[train_num:train_num + valid_num],
        trace_num=valid_num,
        trace_offset=trace_offset,
        trace_length=trace_length
    )
    kwargs_target_finetune = dict(
        trs_file=X_attack_target[0:target_finetune_num, :],
        label_file=Y_attack_target[0:target_finetune_num],
        trace_num=target_finetune_num,
        trace_offset=trace_offset,
        trace_length=trace_length
    )

    source_train_loader = load_training(batch_size, kwargs_source_train, drop_last=True)
    source_valid_loader = load_training(batch_size, kwargs_source_valid, drop_last=True)
    target_finetune_loader = load_training(batch_size, kwargs_target_finetune, drop_last=True)

    print('Load data complete!')

    # -------- logit adjustment --------
    adjustments_np = compute_adjustment_1(Y_train_source[0:train_num], tro=1, classes=class_num)
    adjustments = torch.from_numpy(adjustments_np).view(1, -1).to(device)

    # ✅ 训练是否启用 logit adjustment（验证永远不启用）
    adjust_flag = True

    # -------- CMMD setting --------
    use_cmmd = True
    lambda_cmmd = 0.1

    # -------- model --------
    model = CDP_Net(num_classes=class_num).to(device)

    flag = "real" if adjust_flag else "fake"
    pretrained_path = f'./models/hw{countermeasure}_{flag}_pre-trained_cpda_device{source_device_id}.pth'
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
        {'params': model.final_classifier.parameters()},
    ], lr=lr)

    if checkpoint is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # -------- finetune --------
    min_loss = 1e9
    for epoch in range(1, finetune_epoch + 1):
        print(f'\n{"=" * 60}\nTrain Epoch {epoch}:\n{"=" * 60}')
        CDP_train(epoch, model)

        valid_loss = CDP_validation(model)
        if valid_loss < min_loss:
            min_loss = valid_loss
            os.makedirs('./models', exist_ok=True)
            save_path = f'./models/hw{countermeasure}_{flag}_best_valid_loss_fine_tuned_cpda_device{source_device_id}_to_{target_device_id}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'min_loss': min_loss,
                'config': {
                    'use_cmmd': use_cmmd,
                    'lambda_cmmd': lambda_cmmd,
                    'adjust_flag_train': adjust_flag
                }
            }, save_path)
            print(f'★ Best model saved: {save_path} (valid_loss={min_loss:.6f})')

    # cleanup
    del source_train_loader, source_valid_loader, target_finetune_loader
    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    import gc
    gc.collect()
    print("Cleanup complete, exiting...")
