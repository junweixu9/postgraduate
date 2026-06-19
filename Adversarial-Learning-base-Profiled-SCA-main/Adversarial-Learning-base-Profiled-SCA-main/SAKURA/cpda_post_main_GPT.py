import os
from torch.utils.data import Dataset, DataLoader
import torch
from torch import optim
import numpy as np
from torch import nn
import random

# 【优化】开启 cudnn 自动寻优
torch.backends.cudnn.benchmark = True

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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
        trace_tensor = torch.from_numpy(trace).float().unsqueeze(0)
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


# =========================
# 2) AES tables / HW map
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
# 3) MMD (your original)
# =========================
def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance) / (n_samples ** 2 - n_samples)

    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)


def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss


# =========================
# 4) CORAL (added)
# =========================
def coral_loss(source, target):
    """
    Deep CORAL loss:
    ||C_s - C_t||_F^2 / (4 d^2)
    source: (B, d), target: (B, d)
    """
    d = source.size(1)
    # Centered features
    source = source - source.mean(dim=0, keepdim=True)
    target = target - target.mean(dim=0, keepdim=True)

    # Covariance (unbiased with n-1)
    ns = source.size(0)
    nt = target.size(0)
    # avoid division by 0 if batch size is 1
    denom_s = max(ns - 1, 1)
    denom_t = max(nt - 1, 1)

    cs = (source.t() @ source) / denom_s
    ct = (target.t() @ target) / denom_t

    loss = torch.mean((cs - ct) ** 2)

    return loss


# =========================
# 5) Logit adjustment
# =========================
def compute_adjustment_1(Y_profiling, tro, classes=9, eps=1e-12):
    counts = np.bincount(Y_profiling, minlength=classes).astype(np.float64)
    pi = counts / np.maximum(counts.sum(), 1.0)
    adjustments = np.log(np.power(pi, tro) + eps)  # log(pi^tau)
    return adjustments.astype(np.float32)


# =========================
# 6) The model: MMD + CORAL
# =========================
class CDP_Net(nn.Module):
    def __init__(self, num_classes=9):
        super(CDP_Net, self).__init__()
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

    def forward(self, source, target):
        # ===== source forward =====
        src_feat = self.features(source)
        src_0 = src_feat.view(src_feat.size(0), -1)          # (B, 448)
        src_1 = self.classifier_1(src_0)                     # (B, 2)
        src_logits = self.final_classifier(src_1)            # (B, C)

        # ===== target forward =====
        tgt_feat = self.features(target)
        tgt_0 = tgt_feat.view(tgt_feat.size(0), -1)          # (B, 448)
        tgt_1 = self.classifier_1(tgt_0)                     # (B, 2)

        # ===== MMD on two levels (same as your code) =====
        mmd_loss = mmd_rbf(src_0, tgt_0) + mmd_rbf(src_1, tgt_1)

        # ===== CORAL on two levels (NEW) =====
        coral = coral_loss(src_0, tgt_0) + coral_loss(src_1, tgt_1)

        return src_logits, mmd_loss, coral


# =========================
# 7) Train / Validation
# =========================
def CDP_train(epoch, model):
    model.train()

    iter_source = iter(source_train_loader)
    iter_target = iter(target_finetune_loader)
    num_iter_target = len(target_finetune_loader)

    finetune_trace_all = torch.zeros((num_iter_target, batch_size, 1, trace_length))
    for i in range(num_iter_target):
        data_batch, _ = next(iter_target)
        finetune_trace_all[i, :, :, :] = data_batch

    num_iter = len(source_train_loader)
    clf_criterion = nn.CrossEntropyLoss()

    for i in range(1, num_iter + 1):
        source_data, source_label = next(iter_source)
        target_data = finetune_trace_all[(i - 1) % num_iter_target, :, :, :]

        if cuda:
            source_data = source_data.cuda(non_blocking=True)
            source_label = source_label.cuda(non_blocking=True)
            target_data = target_data.cuda(non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        source_preds, mmd_loss, coral = model(source_data, target_data)

        if adjust_flag:
            source_preds = source_preds + 1 * adjustments

        clf_loss = clf_criterion(source_preds, source_label)

        # ===== MMD-CORAL total =====
        loss = clf_loss + lambda_mmd * mmd_loss + lambda_coral * coral

        loss.backward()
        optimizer.step()

        if i % log_interval == 0:
            print(
                'Train Epoch {}: [{}/{} ({:.0f}%)]\t'
                'total_loss: {:.6f}\tclf_loss: {:.6f}\tmmd_loss: {:.6f}\tcoral_loss: {:.6f}'.format(
                    epoch,
                    i * len(source_data),
                    len(source_train_loader) * batch_size,
                    100. * i / len(source_train_loader),
                    loss.item(),
                    clf_loss.item(),
                    mmd_loss.item(),
                    coral.item()
                )
            )


def CDP_validation(model):
    clf_criterion = nn.CrossEntropyLoss()
    model.eval()

    iter_source = iter(source_valid_loader)
    iter_target = iter(target_finetune_loader)
    num_iter_target = len(target_finetune_loader)

    finetune_trace_all = torch.zeros((num_iter_target, batch_size, 1, trace_length))
    for i in range(num_iter_target):
        data_batch, _ = next(iter_target)
        finetune_trace_all[i, :, :, :] = data_batch

    num_iter = len(source_valid_loader)
    total_clf_loss = 0.0
    total_mmd_loss = 0.0
    total_coral_loss = 0.0
    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for i in range(1, num_iter + 1):
            source_data, source_label = next(iter_source)
            target_data = finetune_trace_all[(i - 1) % num_iter_target, :, :, :]

            if cuda:
                source_data = source_data.cuda(non_blocking=True)
                source_label = source_label.cuda(non_blocking=True)
                target_data = target_data.cuda(non_blocking=True)

            valid_preds, mmd_loss, coral = model(source_data, target_data)
            clf_loss = clf_criterion(valid_preds, source_label)

            loss = clf_loss + lambda_mmd * mmd_loss + lambda_coral * coral

            total_clf_loss += clf_loss.item()
            total_mmd_loss += mmd_loss.item()
            total_coral_loss += coral.item()
            total_loss += loss.item()

            pred = valid_preds.data.max(1)[1]
            correct += pred.eq(source_label.data.view_as(pred)).cpu().sum()

    total_loss /= len(source_valid_loader)
    total_clf_loss /= len(source_valid_loader)
    total_mmd_loss /= len(source_valid_loader)
    total_coral_loss /= len(source_valid_loader)

    print(
        'Validation: total_loss: {:.4f}, clf_loss: {:.4f}, mmd_loss: {:.4f}, coral_loss: {:.4f}, '
        'accuracy: {}/{} ({:.2f}%)'.format(
            total_loss, total_clf_loss, total_mmd_loss, total_coral_loss,
            correct, len(source_valid_loader.dataset),
            100. * correct / len(source_valid_loader.dataset)
        )
    )

    return total_loss


# =========================
# 8) Main
# =========================
if __name__ == '__main__':
    source_device_id = 1
    target_device_id = 3

    labeling_method = 'hw'

    batch_size = 200
    finetune_epoch = 30
    lr = 0.001
    log_interval = 50

    train_num = 85000
    valid_num = 5000
    target_finetune_num = 200

    trace_offset = 0
    trace_length = 1000

    # ===== MMD-CORAL weights =====
    lambda_mmd = 0.05
    lambda_coral = 1   # 【新增】你可以先和 MMD 同量级，后面再调参

    source_file_path = './Data/device1/'
    target_file_path = './Data/device3/'

    no_cuda = False
    cuda = not no_cuda and torch.cuda.is_available()

    seed = 8
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    adjust_flag = False

    X_train_source = np.load(source_file_path + 'X_train.npy')
    Y_train_source = np.load(source_file_path + 'Y_train.npy')
    X_attack_target = np.load(target_file_path + 'X_attack.npy')
    Y_attack_target = np.load(target_file_path + 'Y_attack.npy')

    if labeling_method == 'identity':
        class_num = 256
    elif labeling_method == 'hw':
        class_num = 9
        Y_train_source = calculate_HW(Y_train_source)

    # ===== normalize per-trace (keep your logic) =====
    mn = np.repeat(np.mean(X_train_source, axis=1, keepdims=True), X_train_source.shape[1], axis=1)
    std = np.repeat(np.std(X_train_source, axis=1, keepdims=True), X_train_source.shape[1], axis=1)
    X_train_source = (X_train_source - mn) / std

    mn = np.repeat(np.mean(X_attack_target, axis=1, keepdims=True), X_attack_target.shape[1], axis=1)
    std = np.repeat(np.std(X_attack_target, axis=1, keepdims=True), X_attack_target.shape[1], axis=1)
    X_attack_target = (X_attack_target - mn) / std

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
    # 【小修】validation 建议用 shuffle=False
    source_valid_loader = load_testing(batch_size, kwargs_source_valid)
    target_finetune_loader = load_training(batch_size, kwargs_target_finetune)
    print('Load data complete!')

    # ===== logit adjustment =====
    adjustments_np = compute_adjustment_1(Y_train_source[0:train_num], tro=1, classes=9)
    adjustments = torch.from_numpy(adjustments_np).view(1, -1)
    if cuda:
        adjustments = adjustments.cuda()

    adjust_flag = True

    model = CDP_Net(num_classes=9)

    flag = "real" if adjust_flag else "fake"
    path = ('./models/' + str(flag) + '_pre-trained_cpda_device{}.pth'.format(source_device_id))
    print("Loading model:", path)

    checkpoint = None
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location="cpu")
        model_dict = checkpoint.get('model_state_dict', None)
        if model_dict is not None:
            model.load_state_dict(model_dict)
    else:
        print(f"Warning: Pretrained model not found at {path}")

    optimizer = optim.Adam([
        {'params': model.features.parameters()},
        {'params': model.classifier_1.parameters()},
        {'params': model.final_classifier.parameters()},
    ], lr=lr)

    if cuda:
        model.cuda()

    if checkpoint is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    min_loss = 1e9

    for epoch in range(1, finetune_epoch + 1):
        print(f'Train Epoch {epoch}:')
        CDP_train(epoch, model)

        with torch.no_grad():
            valid_loss = CDP_validation(model)
            if valid_loss < min_loss:
                min_loss = valid_loss
                if not os.path.exists('./models'):
                    os.makedirs('./models')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, './models/' + str(flag) + 'best_valid_loss_fine_tuned_mmd_coral_device_GPT_{}_to_{}.pth'.format(
                    source_device_id, target_device_id
                ))
                print("Saved best checkpoint (min valid loss).")
