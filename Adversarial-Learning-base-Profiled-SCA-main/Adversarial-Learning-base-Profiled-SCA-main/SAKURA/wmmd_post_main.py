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
# Dataset / DataLoader
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
# AES tables (unchanged)
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

InvSbox = [82, 9, 106, 213, 48, 54, 165, 56, 191, 64, 163, 158, 129, 243, 215, 251, 124, 227, 57, 130, 155, 47, 255,
           135, 52, 142, 67, 68, 196, 222, 233, 203, 84, 123, 148, 50, 166, 194, 35, 61, 238, 76, 149, 11, 66, 250, 195, 78,
           8, 46, 161, 102, 40, 217, 36, 178, 118, 91, 162, 73, 109, 139, 209, 37, 114, 248, 246, 100, 134, 104, 152, 22,
           212, 164, 92, 204, 93, 101, 182, 146, 108, 112, 72, 80, 253, 237, 185, 218, 94, 21, 70, 87, 167, 141, 157, 132,
           144, 216, 171, 0, 140, 188, 211, 10, 247, 228, 88, 5, 184, 179, 69, 6, 208, 44, 30, 143, 202, 63, 15, 2, 193, 175,
           189, 3, 1, 19, 138, 107, 58, 145, 17, 65, 79, 103, 220, 234, 151, 242, 207, 206, 240, 180, 230, 115, 150, 172,
           116, 34, 231, 173, 53, 133, 226, 249, 55, 232, 28, 117, 223, 110, 71, 241, 26, 113, 29, 41, 197, 137, 111, 183, 98,
           14, 170, 24, 190, 27, 252, 86, 62, 75, 198, 210, 121, 32, 154, 219, 192, 254, 120, 205, 90, 244, 31, 221, 168, 51,
           136, 7, 199, 49, 177, 18, 16, 89, 39, 128, 236, 95, 96, 81, 127, 169, 25, 181, 74, 13, 45, 229, 122, 159, 147, 201,
           156, 239, 160, 224, 59, 77, 174, 42, 245, 176, 200, 235, 187, 60, 131, 83, 153, 97, 23, 43, 4, 126, 186, 119, 214,
           38, 225, 105, 20, 99, 85, 33, 12, 125]

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
# Models (mostly unchanged)
# =========================
class ATN(nn.Module):
    def __init__(self, num_classes=9):
        super(ATN, self).__init__()
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

    def forward(self, input):
        x = self.features(input)
        feature = x.view(x.size(0), -1)
        output = self.classifier_1(feature)
        output = self.final_classifier(output)
        return feature, output


# =========================
# Kernel / (W)MMD
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


def wmmd_rbf(source, target, source_labels, alpha, kernel_mul=2.0, kernel_num=5, fix_sigma=None, eps=1e-12):
    """
    Weighted MMD (WMMD):
      ms = sum_i w_i phi(x_s_i), where w_i = alpha[y_s_i], normalized s.t. sum w_i = 1
      mt = (1/N) sum_j phi(x_t_j)
      WMMD^2 = <ms,ms> + <mt,mt> - 2<ms,mt>
    alpha: Tensor shape [C], class-wise weights (alpha_c = w_t^c / w_s^c)
    """
    batch_size_s = int(source.size(0))
    batch_size_t = int(target.size(0))

    kernels = guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    K_ss = kernels[:batch_size_s, :batch_size_s]
    K_tt = kernels[batch_size_s:, batch_size_s:]
    K_st = kernels[:batch_size_s, batch_size_s:]

    # per-sample weights from class-wise alpha
    w = alpha[source_labels]  # [Bs]
    w = w.clamp_min(eps)
    w = w / (w.sum() + eps)   # normalize to sum=1

    term_ss = (w.view(-1, 1) * w.view(1, -1) * K_ss).sum()
    term_tt = K_tt.mean()  # 1/N^2 sum
    term_st = (w.view(-1, 1) * K_st).mean(dim=1).sum()  # sum_i w_i * (1/N sum_j k)

    loss = term_ss + term_tt - 2.0 * term_st
    return loss


# =========================
# CDP_Net (modified: forward takes source_label & uses WMMD)
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

    def forward(self, source, target, source_label, alpha):
        # source flow
        source_feat = self.features(source)
        source_0 = source_feat.view(source_feat.size(0), -1)
        source_1 = self.classifier_1(source_0)

        # target flow
        target_feat = self.features(target)
        target_0 = target_feat.view(target_feat.size(0), -1)
        target_1 = self.classifier_1(target_0)

        # WMMD at two layers (like your original mmd at two layers)
        wmmd_loss = 0.0
        wmmd_loss = wmmd_loss + wmmd_rbf(source_0, target_0, source_label, alpha)
        wmmd_loss = wmmd_loss + wmmd_rbf(source_1, target_1, source_label, alpha)

        logits = self.final_classifier(source_1)
        return logits, wmmd_loss


# =========================
# Logit Adjustment (unchanged)
# =========================
def compute_adjustment_1(Y_profiling, tro, classes=9, eps=1e-12):
    counts = np.bincount(Y_profiling, minlength=classes).astype(np.float64)
    pi = counts / np.maximum(counts.sum(), 1.0)
    adjustments = np.log(np.power(pi, tro) + eps)
    return adjustments.astype(np.float32)


# =========================
# WMMD alpha estimation (NEW)
# =========================
@torch.no_grad()
def estimate_alpha_from_pseudolabels(model, target_loader, source_labels_np, num_classes, cuda, eps=1e-12,
                                    clamp_min=0.1, clamp_max=10.0):
    """
    Estimate alpha_c = w_t^c / w_s^c
    - w_s^c from source labels (true)
    - w_t^c from pseudo labels on target (argmax softmax)
    """
    # source prior
    ws_counts = np.bincount(source_labels_np.astype(int), minlength=num_classes).astype(np.float64)
    ws = ws_counts / np.maximum(ws_counts.sum(), 1.0)

    # target pseudo prior
    wt_counts = np.zeros(num_classes, dtype=np.float64)

    model.eval()
    for xb, _ in target_loader:  # ignore target true labels (UDA-style)
        if cuda:
            xb = xb.cuda(non_blocking=True)
        _, logits = model.features(xb), None  # not used, but keep style consistent
        # run full forward for logits: we can reuse ATN-style but here CDP_Net needs target+source
        # So we do a simple classifier pass: features->classifier_1->final_classifier
        feat = model.features(xb).view(xb.size(0), -1)
        h = model.classifier_1(feat)
        out = model.final_classifier(h)
        pseudo = out.argmax(dim=1).detach().cpu().numpy()
        wt_counts += np.bincount(pseudo, minlength=num_classes)

    wt = wt_counts / np.maximum(wt_counts.sum(), 1.0)

    alpha = wt / (ws + eps)
    alpha = np.clip(alpha, clamp_min, clamp_max).astype(np.float32)

    alpha_t = torch.from_numpy(alpha)
    if cuda:
        alpha_t = alpha_t.cuda()
    return alpha_t


# =========================
# Train / Validation
# =========================
def CDP_train(epoch, model, alpha):
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

        # predictions + WMMD
        source_preds, wmmd_loss = model(source_data, target_data, source_label, alpha)

        # logit adjustment (your original)
        if adjust_flag:
            source_preds = source_preds + 1.0 * adjustments

        clf_loss = clf_criterion(source_preds, source_label)
        loss = clf_loss + lambda_ * wmmd_loss

        loss.backward()
        optimizer.step()

        if i % log_interval == 0:
            print('Train Epoch {}: [{}/{} ({:.0f}%)]\ttotal_loss: {:.6f}\tclf_loss: {:.6f}\twmmd_loss: {:.6f}'.format(
                epoch, i * len(source_data), len(source_train_loader) * batch_size,
                100. * i / len(source_train_loader), loss.item(), clf_loss.item(), wmmd_loss.item()))


def CDP_validation(model, alpha):
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
    total_wmmd_loss = 0.0
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

            valid_preds, wmmd_loss = model(source_data, target_data, source_label, alpha)

            clf_loss = clf_criterion(valid_preds, source_label)
            loss = clf_loss + lambda_ * wmmd_loss

            total_clf_loss += clf_loss.item()
            total_wmmd_loss += wmmd_loss.item()
            total_loss += loss.item()

            pred = valid_preds.data.max(1)[1]
            correct += pred.eq(source_label.data.view_as(pred)).cpu().sum()

    total_loss /= len(source_valid_loader)
    total_clf_loss /= len(source_valid_loader)
    total_wmmd_loss /= len(source_valid_loader)

    print('Validation: total_loss: {:.4f}, clf_loss: {:.4f}, wmmd_loss: {:.4f}, accuracy: {}/{} ({:.2f}%)'.format(
        total_loss, total_clf_loss, total_wmmd_loss, correct, len(source_valid_loader.dataset),
        100. * correct / len(source_valid_loader.dataset)))

    return total_loss


# =========================
# Main
# =========================
if __name__ == '__main__':
    source_device_id = 1
    target_device_id = 2

    real_key_01 = 0x21
    real_key_02 = 0xCD

    labeling_method = 'hw'
    _lambda = 0.05
    batch_size = 200
    total_epoch = 100
    finetune_epoch = 30
    lr = 0.001
    log_interval = 50

    train_num = 85000
    valid_num = 5000
    source_test_num = 9900
    target_finetune_num = 200
    target_test_num = 9400

    trace_offset = 0
    lambda_ = 0.05
    trace_length = 1000

    source_file_path = './Data/device1/'
    target_file_path = './Data/device3/'

    no_cuda = False
    cuda = not no_cuda and torch.cuda.is_available()

    seed = 8
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    X_train_source = np.load(source_file_path + 'X_train.npy')
    Y_train_source = np.load(source_file_path + 'Y_train.npy')
    X_attack_target = np.load(target_file_path + 'X_attack.npy')
    Y_attack_target = np.load(target_file_path + 'Y_attack.npy')

    if labeling_method == 'identity':
        class_num = 256
    elif labeling_method == 'hw':
        class_num = 9
        Y_train_source = calculate_HW(Y_train_source)

    # normalize
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
    source_valid_loader = load_training(batch_size, kwargs_source_valid)
    target_finetune_loader = load_training(batch_size, kwargs_target_finetune)
    print('Load data complete!')

    # logit adjustment
    adjustments_np = compute_adjustment_1(Y_train_source[0:train_num], tro=1, classes=class_num)
    adjustments = torch.from_numpy(adjustments_np).view(1, -1)
    if cuda:
        adjustments = adjustments.cuda()
    adjust_flag = True

    model = CDP_Net(num_classes=class_num)

    flag = "real" if adjust_flag else "fake"
    path = ('./models/' + str(flag) + '_pre-trained_cpda_device{}.pth'.format(source_device_id))
    print("Loading model:", path)

    checkpoint = None
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location="cpu")
        model_dict = checkpoint['model_state_dict']
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

    # source labels (for w_s) as numpy
    source_labels_np_for_ws = Y_train_source[0:train_num].copy()

    for epoch in range(1, finetune_epoch + 1):
        print(f'\n========== Epoch {epoch} ==========')

        # ====== WMMD: estimate alpha from pseudo labels on target ======
        alpha = estimate_alpha_from_pseudolabels(
            model=model,
            target_loader=target_finetune_loader,
            source_labels_np=source_labels_np_for_ws,
            num_classes=class_num,
            cuda=cuda,
            clamp_min=0.1,
            clamp_max=10.0
        )
        if cuda:
            alpha = alpha.cuda()
        print("WMMD alpha (first 9):", alpha.detach().cpu().numpy()[:min(9, class_num)])

        # train
        CDP_train(epoch, model, alpha)

        # validation
        with torch.no_grad():
            valid_loss = CDP_validation(model, alpha)
            if valid_loss < min_loss:
                min_loss = valid_loss
                if not os.path.exists('./models'):
                    os.makedirs('./models')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, './models/' + str(flag) + '_wmdd_best_valid_loss_fine_tuned_cpda_device{}_to_{}.pth'.format(
                    source_device_id, target_device_id))
                print("Saved best checkpoint (min_loss updated).")
