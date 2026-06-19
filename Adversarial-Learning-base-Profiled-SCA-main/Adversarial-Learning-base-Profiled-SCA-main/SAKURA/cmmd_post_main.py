import gc
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
# 0) Seed
# =========================
def seed_everything(seed: int = 8):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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

InvSbox = [82, 9, 106, 213, 48, 54, 165, 56, 191, 64, 163, 158, 129, 243, 215, 251, 124, 227, 57, 130, 155, 47, 255,
           135,
           52, 142, 67, 68, 196, 222, 233, 203, 84, 123, 148, 50, 166, 194, 35, 61, 238, 76, 149, 11, 66, 250, 195, 78,
           8,
           46, 161, 102, 40, 217, 36, 178, 118, 91, 162, 73, 109, 139, 209, 37, 114, 248, 246, 100, 134, 104, 152, 22,
           212,
           164, 92, 204, 93, 101, 182, 146, 108, 112, 72, 80, 253, 237, 185, 218, 94, 21, 70, 87, 167, 141, 157, 132,
           144,
           216, 171, 0, 140, 188, 211, 10, 247, 228, 88, 5, 184, 179, 69, 6, 208, 44, 30, 143, 202, 63, 15, 2, 193, 175,
           189,
           3, 1, 19, 138, 107, 58, 145, 17, 65, 79, 103, 220, 234, 151, 242, 207, 206, 240, 180, 230, 115, 150, 172,
           116, 34,
           231, 173, 53, 133, 226, 249, 55, 232, 28, 117, 223, 110, 71, 241, 26, 113, 29, 41, 197, 137, 111, 183, 98,
           14, 170,
           24, 190, 27, 252, 86, 62, 75, 198, 210, 121, 32, 154, 219, 192, 254, 120, 205, 90, 244, 31, 221, 168, 51,
           136, 7,
           199, 49, 177, 18, 16, 89, 39, 128, 236, 95, 96, 81, 127, 169, 25, 181, 74, 13, 45, 229, 122, 159, 147, 201,
           156,
           239, 160, 224, 59, 77, 174, 42, 245, 176, 200, 235, 187, 60, 131, 83, 153, 97, 23, 43, 4, 126, 186, 119, 214,
           38,
           225, 105, 20, 99, 85, 33, 12, 125]

HW_byte = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2,
           3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2, 3,
           3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3,
           4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4,
           3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5,
           6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4,
           4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5,
           6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8]

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
        trace_tensor = torch.from_numpy(trace).float().unsqueeze(0)  # [1, L]
        label_tensor = torch.tensor(self.label_file[index], dtype=torch.long)
        return trace_tensor, label_tensor

    def __len__(self):
        return self.trace_num


def load_training(batch_size, kwargs, drop_last=True):
    data = TorchDataset(**kwargs)
    train_loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    return train_loader


def load_testing(batch_size, kwargs, drop_last=True):
    data = TorchDataset(**kwargs)
    test_loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=False,
        drop_last=drop_last,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    return test_loader


# =========================
# 2) Kernel + CMMD
# =========================
def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    Multi-kernel RBF
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
    CMMD using label-consistency masks (one-hot):
    E[S ⊙ K_ss] + E[T ⊙ K_tt] - 2 E[ST ⊙ K_st]
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

    S = s_onehot @ s_onehot.t()      # [bs,bs]
    T = t_onehot @ t_onehot.t()      # [bt,bt]
    ST = s_onehot @ t_onehot.t()     # [bs,bt]

    return torch.mean(S * XX + T * YY - 2.0 * ST * XY)


# =========================
# 3) Model (extract feats + logits)
# =========================
class CDP_Net(nn.Module):
    def __init__(self, num_classes=9):
        super(CDP_Net, self).__init__()
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=1),
            nn.SELU(),
            nn.BatchNorm1d(8),
            nn.AvgPool1d(kernel_size=2, stride=2),

            nn.Conv1d(8, 16, kernel_size=11),
            nn.SELU(),
            nn.BatchNorm1d(16),
            nn.AvgPool1d(kernel_size=11, stride=11),

            nn.Conv1d(16, 32, kernel_size=2),
            nn.SELU(),
            nn.BatchNorm1d(32),
            nn.AvgPool1d(kernel_size=3, stride=3),

            nn.Flatten()
        )

        self.classifier_1 = nn.Sequential(
            nn.Linear(448, 2),
            nn.SELU(),
        )

        self.final_classifier = nn.Sequential(
            nn.Linear(2, num_classes)
        )

    def extract_feats(self, x):
        f0 = self.features(x).view(x.size(0), -1)  # [b, 448]
        f1 = self.classifier_1(f0)                 # [b, 2]
        return f0, f1

    def forward_logits(self, x):
        _, f1 = self.extract_feats(x)
        logits = self.final_classifier(f1)
        return logits


# =========================
# 4) Logit adjustment
# =========================
def compute_adjustment_1(Y_profiling, tro, classes=9, eps=1e-12):
    counts = np.bincount(Y_profiling, minlength=classes).astype(np.float64)
    pi = counts / np.maximum(counts.sum(), 1.0)
    adjustments = np.log(np.power(pi, tro) + eps)  # log(pi^tau)
    return adjustments.astype(np.float32)


# =========================
# 5) Pseudo labels for target (UDA-style)
# =========================
@torch.no_grad()
def get_target_pseudo_labels(model, target_loader, device):
    model.eval()
    pseudo_list = []
    for xb, _ in target_loader:
        xb = xb.to(device, non_blocking=True)
        logits = model.forward_logits(xb)
        pseudo = logits.argmax(dim=1)
        pseudo_list.append(pseudo.cpu())
    return pseudo_list


# =========================
# 6) Train / Validation (CMMD + logit adjustment)
# =========================
def CDP_train(epoch, model):
    model.eval()  # ✅ must be train() because of BatchNorm

    clf_criterion = nn.CrossEntropyLoss()

    # --- target labels source: pseudo or true ---
    # 如果你想用“真标签”做 CMMD（半监督/有标签目标域），改成 True
    use_true_target_label = False

    if not use_true_target_label:
        target_label_list = get_target_pseudo_labels(model, target_finetune_loader, device)
    else:
        # 直接从 loader 取真实标签
        target_label_list = [yb for _, yb in target_finetune_loader]

    iter_source = iter(source_train_loader)
    iter_target = iter(target_finetune_loader)
    num_iter_target = len(target_finetune_loader)

    # ✅ 不再写死 shape，直接预取成 list（更稳）
    finetune_trace_all = []
    for _ in range(num_iter_target):
        xb, _ = next(iter_target)
        finetune_trace_all.append(xb)

    num_iter = len(source_train_loader)

    for i in range(1, num_iter + 1):
        source_data, source_label = next(iter_source)
        t_idx = (i - 1) % num_iter_target
        target_data = finetune_trace_all[t_idx]
        target_label = target_label_list[t_idx]

        source_data = source_data.to(device, non_blocking=True)
        source_label = source_label.to(device, non_blocking=True)
        target_data = target_data.to(device, non_blocking=True)
        target_label = target_label.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # forward
        s0, s1 = model.extract_feats(source_data)
        t0, t1 = model.extract_feats(target_data)

        logits = model.final_classifier(s1)

        if adjust_flag:
            logits = logits + adjustments  # adjustments: [1, C]

        clf_loss = clf_criterion(logits, source_label)

        # ✅ CMMD on two levels (like your original two-level MMD)
        cmmd_0 = cmmd_loss(s0, t0, source_label, target_label, num_classes=class_num)
        cmmd_1 = cmmd_loss(s1, t1, source_label, target_label, num_classes=class_num)
        cmmd = cmmd_0 + cmmd_1

        loss = clf_loss + lambda_ * cmmd
        loss.backward()
        optimizer.step()

        if i % log_interval == 0:
            print(
                'Train Epoch {}: [{}/{} ({:.0f}%)]\t'
                'total_loss: {:.6f}\tclf_loss: {:.6f}\tcmmd_loss: {:.6f}'.format(
                    epoch,
                    i * len(source_data),
                    len(source_train_loader) * source_data.size(0),
                    100. * i / len(source_train_loader),
                    loss.item(), clf_loss.item(), cmmd.item()
                )
            )


@torch.no_grad()
def CDP_validation(model):
    model.eval()
    clf_criterion = nn.CrossEntropyLoss()

    # 与训练一致：默认用伪标签（也可切换成真标签）
    use_true_target_label = False
    if not use_true_target_label:
        target_label_list = get_target_pseudo_labels(model, target_finetune_loader, device)
    else:
        target_label_list = [yb for _, yb in target_finetune_loader]

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
        t_idx = (i - 1) % num_iter_target
        target_data = finetune_trace_all[t_idx]
        target_label = target_label_list[t_idx]

        source_data = source_data.to(device, non_blocking=True)
        source_label = source_label.to(device, non_blocking=True)
        target_data = target_data.to(device, non_blocking=True)
        target_label = target_label.to(device, non_blocking=True)

        s0, s1 = model.extract_feats(source_data)
        t0, t1 = model.extract_feats(target_data)

        logits = model.final_classifier(s1)
        clf_loss = clf_criterion(logits, source_label)

        cmmd_0 = cmmd_loss(s0, t0, source_label, target_label, num_classes=class_num)
        cmmd_1 = cmmd_loss(s1, t1, source_label, target_label, num_classes=class_num)
        cmmd = cmmd_0 + cmmd_1

        loss = clf_loss + lambda_ * cmmd

        total_loss += loss.item()
        total_clf_loss += clf_loss.item()
        total_cmmd_loss += cmmd.item()

        pred = logits.argmax(dim=1)
        correct += (pred == source_label).sum().item()

    n = len(source_valid_loader)
    total_loss /= n
    total_clf_loss /= n
    total_cmmd_loss /= n

    acc = 100.0 * correct / len(source_valid_loader.dataset)
    print(
        'Validation: total_loss: {:.4f}, clf_loss: {:.4f}, cmmd_loss: {:.4f}, '
        'accuracy: {}/{} ({:.2f}%)'.format(
            total_loss, total_clf_loss, total_cmmd_loss,
            correct, len(source_valid_loader.dataset), acc
        )
    )
    return total_loss

# 【优化】将 HW_byte 转换为 Numpy 数组，支持向量化索引
HW_byte_np = np.array(HW_byte, dtype=np.int32)

def calculate_HW(data):
    # 【优化】使用向量化索引代替列表推导式，速度极快
    return HW_byte_np[data.astype(int)]

# =========================
# 7) Main (keep your original)
# =========================
if __name__ == '__main__':
    seed_everything(8)

    source_device_id = 1
    target_device_id = 2

    labeling_method = 'hw'
    batch_size = 200
    finetune_epoch = 15
    lr = 0.001
    log_interval = 50

    train_num = 85000
    valid_num = 5000
    target_finetune_num = 200

    trace_offset = 0
    trace_length = 1000

    lambda_ = 0.05  # CMMD penalty coeff

    source_file_path = './Data/device1/'
    target_file_path = './Data/device2/'

    no_cuda = False
    cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    X_train_source = np.load(source_file_path + 'X_train.npy')
    Y_train_source = np.load(source_file_path + 'Y_train.npy')
    X_attack_target = np.load(target_file_path + 'X_attack.npy')
    Y_attack_target = np.load(target_file_path + 'Y_attack.npy')

    # --- HW labels ---
    # 你原脚本里 calculate_HW / HW_byte_np 在此略（保持你原有的即可）
    if labeling_method == 'identity':
        class_num = 256
    elif labeling_method == 'hw':
        class_num = 9
        # 这里假设你已经把 calculate_HW 定义好了
        Y_train_source = calculate_HW(Y_train_source)
        Y_attack_target = calculate_HW(Y_attack_target)
        pass

    # normalize per-trace
    mn = np.repeat(np.mean(X_train_source, axis=1, keepdims=True), X_train_source.shape[1], axis=1)
    std = np.repeat(np.std(X_train_source, axis=1, keepdims=True), X_train_source.shape[1], axis=1)
    X_train_source = (X_train_source - mn) / (std + 1e-12)

    mn = np.repeat(np.mean(X_attack_target, axis=1, keepdims=True), X_attack_target.shape[1], axis=1)
    std = np.repeat(np.std(X_attack_target, axis=1, keepdims=True), X_attack_target.shape[1], axis=1)
    X_attack_target = (X_attack_target - mn) / (std + 1e-12)

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

    source_train_loader = load_training(batch_size, kwargs_source_train, drop_last=True)
    source_valid_loader = load_training(batch_size, kwargs_source_valid, drop_last=True)
    target_finetune_loader = load_training(batch_size, kwargs_target_finetune, drop_last=True)

    print('Load data complete!')

    # logit adjustment
    adjustments_np = compute_adjustment_1(Y_train_source[0:train_num], tro=1, classes=class_num)
    adjustments = torch.from_numpy(adjustments_np).view(1, -1).to(device)  # float32

    model = CDP_Net(num_classes=class_num).to(device)
    adjust_flag = False  # ✅ 想要 logit adjustment 就 True
    # load pretrained
    flag = "real" if adjust_flag else "fake"
    path = ('./models/' + str(flag) + '_pre-trained_cpda_device{}.pth'.format(source_device_id))
    print("Loading model:", path)
    checkpoint = None
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"Warning: Pretrained model not found at {path}")

    optimizer = optim.Adam([
        {'params': model.features.parameters()},
        {'params': model.classifier_1.parameters()},
        {'params': model.final_classifier.parameters()},
    ], lr=lr)

    if checkpoint is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    min_loss = 1e9
    for epoch in range(1, finetune_epoch + 1):
        print(f'Train Epoch {epoch}:')
        CDP_train(epoch, model)

        valid_loss = CDP_validation(model)
        if valid_loss < min_loss:
            min_loss = valid_loss
            if not os.path.exists('./models'):
                os.makedirs('./models')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, './models/' + str(flag) + '_best_valid_loss_cmmd_finetuned_device{}_to_{}.pth'.format(source_device_id, target_device_id))
            print(f'★ Best saved: {min_loss:.6f}')

    del source_train_loader, source_valid_loader, target_finetune_loader
    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    print("Cleanup complete, exiting...")
