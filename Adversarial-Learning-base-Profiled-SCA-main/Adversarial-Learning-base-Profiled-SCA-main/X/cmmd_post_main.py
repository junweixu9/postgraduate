import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import gc
import random
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.benchmark = True


# =========================
# 0) Reproducibility
# =========================
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
        index = i % self.trace_num
        trace = self.trs_file[index, self.trace_offset: self.trace_offset + self.trace_length]
        x = torch.from_numpy(trace).float().unsqueeze(0)  # [1, L]
        y = torch.tensor(self.label_file[index], dtype=torch.long)
        return x, y

    def __len__(self):
        return self.trace_num


def load_training(batch_size, kwargs, drop_last=True):
    data = TorchDataset(**kwargs)
    loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    return loader


def load_testing(batch_size, kwargs, drop_last=True):
    data = TorchDataset(**kwargs)
    loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=False,
        drop_last=drop_last,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    return loader


# =========================
# 2) CMMD (Conditional MMD)
# =========================
def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    标准多核RBF高斯核
    """
    n_samples = int(source.size(0)) + int(target.size(0))
    total = torch.cat([source, target], dim=0)  # [n, d]

    total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
    total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
    L2_distance = ((total0 - total1) ** 2).sum(2)  # [n, n]

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
    ✅ 兼容 source/target batch_size 不一致的 CMMD

    source: [bs, d]
    target: [bt, d]
    s_label: [bs]
    t_label: [bt]  (pseudo labels)
    """
    bs = source.size(0)
    bt = target.size(0)

    # one-hot
    s_onehot = torch.zeros(bs, num_classes, device=source.device)
    s_onehot.scatter_(1, s_label.view(-1, 1), 1)

    t_onehot = torch.zeros(bt, num_classes, device=target.device)
    t_onehot.scatter_(1, t_label.view(-1, 1), 1)

    kernels = gaussian_kernel(source, target, kernel_mul, kernel_num, fix_sigma)  # [(bs+bt), (bs+bt)]

    XX = kernels[:bs, :bs]               # [bs, bs]
    YY = kernels[bs:bs+bt, bs:bs+bt]     # [bt, bt]
    XY = kernels[:bs, bs:bs+bt]          # [bs, bt]

    # class-consistency masks
    S = s_onehot @ s_onehot.t()          # [bs, bs]
    T = t_onehot @ t_onehot.t()          # [bt, bt]
    ST = s_onehot @ t_onehot.t()         # [bs, bt]

    loss = torch.mean(S * XX + T * YY - 2.0 * ST * XY)
    return loss


# =========================
# 3) Model (CMMD only)
# =========================
class CDP_Net(nn.Module):
    def __init__(self, num_classes=9):
        super(CDP_Net, self).__init__()
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=1),
            nn.SELU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(16, 32, kernel_size=50),
            nn.SELU(),
            nn.AvgPool1d(kernel_size=50, stride=50),
            nn.Conv1d(32, 64, kernel_size=3),
            nn.SELU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        self.classifier_1 = nn.Sequential(
            nn.Linear(64, 20),
            nn.ReLU(inplace=True),
        )
        self.final_classifier = nn.Sequential(
            nn.Linear(20, num_classes)
        )

    def extract_two_level_feats(self, x):
        f = self.features(x)
        f0 = f.view(f.size(0), -1)      # [b, 64]
        f1 = self.classifier_1(f0)      # [b, 20]
        return f0, f1

    def forward(self, source, target, source_label=None, target_label=None, use_cmmd=False):
        """
        返回：logits（source分类），losses字典（cmmd）
        """
        s0, s1 = self.extract_two_level_feats(source)
        t0, t1 = self.extract_two_level_feats(target)

        losses = {}
        if use_cmmd and source_label is not None and target_label is not None:
            cmmd_0 = cmmd_loss(s0, t0, source_label, target_label, num_classes=self.num_classes)
            cmmd_1 = cmmd_loss(s1, t1, source_label, target_label, num_classes=self.num_classes)
            losses["cmmd"] = cmmd_0 + cmmd_1
        else:
            losses["cmmd"] = torch.tensor(0.0, device=source.device)

        logits = self.final_classifier(s1)
        return logits, losses


# =========================
# 4) Logit adjustment
# =========================
def compute_adjustment_1(Y_profiling, tro, classes=9, eps=1e-12):
    counts = np.bincount(Y_profiling, minlength=classes).astype(np.float64)
    pi = counts / np.maximum(counts.sum(), 1.0)
    adj = np.log(np.power(pi, tro) + eps)
    return adj.astype(np.float32)


# =========================
# 5) Pseudo labels
# =========================
@torch.no_grad()
def get_target_pseudo_labels(model, target_loader, device):
    model.eval()
    pseudo_labels_list = []
    for data, _ in target_loader:
        data = data.to(device, non_blocking=True)
        _, f1 = model.extract_two_level_feats(data)
        logits = model.final_classifier(f1)
        pseudo = logits.argmax(dim=1)
        pseudo_labels_list.append(pseudo.cpu())
    return pseudo_labels_list


# =========================
# 6) Train / Validation (CMMD + logit adjustment)
# =========================
def CDP_train(epoch, model,
              use_cmmd=True, lambda_cmmd=0.1,
              alpha=0.5, adjust_flag=True, adjustments=None):
    model.eval()
    clf_criterion = nn.CrossEntropyLoss()

    # pseudo labels（用当前模型）
    if use_cmmd:
        print("Computing pseudo labels for target domain...")
        target_pseudo_labels_list = get_target_pseudo_labels(model, target_finetune_loader, device)
    else:
        target_pseudo_labels_list = None

    iter_source = iter(source_train_loader)
    iter_target = iter(target_finetune_loader)
    num_iter_target = len(target_finetune_loader)

    # 预取target batch（CPU）
    finetune_trace_all = []
    for _ in range(num_iter_target):
        data_batch, _ = next(iter_target)
        finetune_trace_all.append(data_batch)

    num_iter = len(source_train_loader)

    for i in range(1, num_iter + 1):
        source_data, source_label = next(iter_source)
        target_idx = (i - 1) % num_iter_target
        target_data = finetune_trace_all[target_idx]

        if use_cmmd:
            target_label = target_pseudo_labels_list[target_idx]
        else:
            target_label = None

        # move to device
        source_data = source_data.to(device, non_blocking=True)
        source_label = source_label.to(device, non_blocking=True)
        target_data = target_data.to(device, non_blocking=True)
        if target_label is not None:
            target_label = target_label.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits, losses = model(
            source_data, target_data,
            source_label=source_label,
            target_label=target_label,
            use_cmmd=use_cmmd
        )

        # ✅ 训练阶段启用 logit adjustment
        if adjust_flag and adjustments is not None:
            logits = logits + adjustments

        clf_loss = clf_criterion(logits, source_label)

        total_loss = alpha * clf_loss
        if use_cmmd:
            total_loss = total_loss + lambda_cmmd * losses["cmmd"]

        total_loss.backward()
        optimizer.step()

        if i % log_interval == 0:
            print(
                'Train Epoch {}: [{}/{} ({:.0f}%)]\t'
                'total_loss: {:.6f}\tclf_loss: {:.6f}\tcmmd: {:.6f}'.format(
                    epoch,
                    i * len(source_data),
                    len(source_train_loader.dataset),
                    100. * i / len(source_train_loader),
                    total_loss.item(),
                    clf_loss.item(),
                    losses["cmmd"].item() if use_cmmd else 0.0
                )
            )


@torch.no_grad()
def CDP_validation(model,
                   use_cmmd=True, lambda_cmmd=0.1,
                   alpha=0.5):
    """
    ✅ 验证阶段：严格不使用 logit adjustment（与你原设定一致）
    """
    model.eval()
    clf_criterion = nn.CrossEntropyLoss()

    if use_cmmd:
        target_pseudo_labels_list = get_target_pseudo_labels(model, target_finetune_loader, device)
    else:
        target_pseudo_labels_list = None

    iter_source = iter(source_valid_loader)
    iter_target = iter(target_finetune_loader)
    num_iter_target = len(target_finetune_loader)

    finetune_trace_all = []
    for _ in range(num_iter_target):
        data_batch, _ = next(iter_target)
        finetune_trace_all.append(data_batch)

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

        logits, losses = model(
            source_data, target_data,
            source_label=source_label,
            target_label=target_label,
            use_cmmd=use_cmmd
        )

        clf_loss = clf_criterion(logits, source_label)

        loss = alpha * clf_loss
        if use_cmmd:
            loss = loss + lambda_cmmd * losses["cmmd"]

        total_loss += loss.item()
        total_clf_loss += clf_loss.item()
        total_cmmd_loss += losses["cmmd"].item() if use_cmmd else 0.0

        pred = logits.data.max(1)[1]
        correct += pred.eq(source_label.data.view_as(pred)).sum().item()

    n = len(source_valid_loader)
    total_loss /= n
    total_clf_loss /= n
    total_cmmd_loss /= n

    acc = 100.0 * correct / len(source_valid_loader.dataset)
    print(
        'Validation: total_loss: {:.4f}, clf_loss: {:.4f}, cmmd: {:.4f}, '
        'accuracy: {}/{} ({:.2f}%)'.format(
            total_loss, total_clf_loss, total_cmmd_loss,
            correct, len(source_valid_loader.dataset), acc
        )
    )
    return total_loss


# =========================
# 7) Main
# =========================
if __name__ == '__main__':
    seed_everything(8)

    DEVICE_CONFIG = {i: {'key': i, 'folder': f'device{i:02d}'} for i in range(1, 9)}
    source_device_id = 1
    target_device_id = 5

    source_file_path = f"./Data/{DEVICE_CONFIG[source_device_id]['folder']}/"
    target_file_path = f"./Data/{DEVICE_CONFIG[target_device_id]['folder']}/"

    # -----------------
    # hyperparams
    # -----------------
    batch_size = 50
    finetune_epoch = 30
    lr = 0.001
    log_interval = 40

    train_num = 20000
    valid_num = 5000
    source_test_num = 5000
    target_finetune_num = 50
    target_test_num = 4500

    trace_offset = 0
    trace_length = 500

    # 分类损失权重
    alpha = 1

    # 域适应：只用 CMMD
    use_cmmd = True
    lambda_cmmd = 0.1

    no_cuda = False
    cuda = (not no_cuda) and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    # -----------------
    # load data
    # -----------------
    X_train_source = np.load(source_file_path + 'X_train.npy')
    Y_train_source = np.load(source_file_path + 'Y_train.npy')
    X_attack_source = np.load(source_file_path + 'X_attack.npy')
    Y_attack_source = np.load(source_file_path + 'Y_attack.npy')
    X_attack_target = np.load(target_file_path + 'X_attack.npy')
    Y_attack_target = np.load(target_file_path + 'Y_attack.npy')

    def hstd(X, ref=None):
        if ref is None:
            ref = X
        mn = np.repeat(np.mean(ref, axis=1, keepdims=True), X.shape[1], axis=1)
        std = np.repeat(np.std(ref, axis=1, keepdims=True), X.shape[1], axis=1)
        return (X - mn) / (std + 1e-12)

    X_train_source = hstd(X_train_source, X_train_source)
    X_attack_source = hstd(X_attack_source, X_attack_source)
    X_attack_target = hstd(X_attack_target, X_attack_target)

    # logit adjustment
    adjustments_np = compute_adjustment_1(Y_train_source[0:train_num], tro=0.5, classes=9)
    adjustments = torch.from_numpy(adjustments_np).view(1, -1).to(device)

    # -----------------
    # dataloaders
    # -----------------
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
    kwargs_source_test = {
        'trs_file': X_attack_source,
        'label_file': Y_attack_source,
        'trace_num': source_test_num,
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
    kwargs_target = {
        'trs_file': X_attack_target[target_finetune_num:target_finetune_num + target_test_num, :],
        'label_file': Y_attack_target[target_finetune_num:target_finetune_num + target_test_num],
        'trace_num': target_test_num,
        'trace_offset': trace_offset,
        'trace_length': trace_length,
    }

    source_train_loader = load_training(batch_size, kwargs_source_train, drop_last=True)
    source_valid_loader = load_training(batch_size, kwargs_source_valid, drop_last=True)
    source_test_loader = load_testing(batch_size, kwargs_source_test, drop_last=True)

    # finetune loader：drop_last=False 也OK（我已让 CMMD 支持不同 batch size）
    target_finetune_loader = load_training(batch_size, kwargs_target_finetune, drop_last=False)
    target_test_loader = load_testing(batch_size, kwargs_target, drop_last=True)

    print('Load data complete!')
    print("len(source_train_loader) =", len(source_train_loader))
    print("len(target_finetune_loader) =", len(target_finetune_loader))
    print("\n=== Domain Adaptation Configuration ===")
    print(f"Use CMMD: {use_cmmd} (λ={lambda_cmmd})")
    print(f"Classification loss weight (α): {alpha}")
    print("=" * 40 + "\n")

    # -----------------
    # model & optimizer
    # -----------------
    CDP_model = CDP_Net(num_classes=9).to(device)
    print('Construct model complete')

    adjust_flag = True  # 训练阶段是否启用logit adjustment
    flag = "real" if adjust_flag else "fake"

    if flag == "real":
        checkpoint = torch.load('./models/True_best_pre-trained_device1.pth', map_location=device)
        CDP_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        checkpoint = torch.load('./models/fake_pre-trained_cpda_device1.pth', map_location=device)
        CDP_model.load_state_dict(checkpoint['model_state_dict'])

    optimizer = optim.Adam([
        {'params': CDP_model.features.parameters()},
        {'params': CDP_model.classifier_1.parameters()},
        {'params': CDP_model.final_classifier.parameters()}
    ], lr=lr)

    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # -----------------
    # finetune
    # -----------------
    min_loss = 1e9
    for epoch in range(1, finetune_epoch + 1):
        print(f'\n{"=" * 60}')
        print(f'Train Epoch {epoch}:')
        print(f'{"=" * 60}')

        CDP_train(
            epoch, CDP_model,
            use_cmmd=use_cmmd,
            lambda_cmmd=lambda_cmmd,
            alpha=alpha,
            adjust_flag=adjust_flag,
            adjustments=adjustments
        )

        valid_loss = CDP_validation(
            CDP_model,
            use_cmmd=use_cmmd,
            lambda_cmmd=lambda_cmmd,
            alpha=alpha
        )

        if valid_loss < min_loss:
            min_loss = valid_loss
            save_name = f'./models/cmmd_real_best_valid_loss_finetuned_device{source_device_id}_to_{target_device_id}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': CDP_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'min_loss': min_loss,
                'config': {
                    'use_cmmd': use_cmmd,
                    'lambda_cmmd': lambda_cmmd,
                    'alpha': alpha,
                    'adjust_flag_train': adjust_flag
                }
            }, save_name)
            print(f'★ Best model saved with validation loss: {min_loss:.6f}')

    # cleanup
    del source_train_loader
    del source_valid_loader
    del target_finetune_loader

    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    gc.collect()
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
