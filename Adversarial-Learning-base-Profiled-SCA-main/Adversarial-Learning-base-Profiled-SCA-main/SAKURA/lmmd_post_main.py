import os
import time

from torch.utils.data import Dataset
import torch
from torch import optim
import numpy as np
from torch import nn
import random

from torchsummary import summary

# 【优化】开启 cudnn 自动寻优
torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# =========================
# 1) Dataset / DataLoader
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
# 2) AES tables (原样保留)
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
           135, 52, 142, 67, 68, 196, 222, 233, 203, 84, 123, 148, 50, 166, 194, 35, 61, 238, 76, 149, 11, 66, 250, 195,
           78,
           8, 46, 161, 102, 40, 217, 36, 178, 118, 91, 162, 73, 109, 139, 209, 37, 114, 248, 246, 100, 134, 104, 152,
           22,
           212, 164, 92, 204, 93, 101, 182, 146, 108, 112, 72, 80, 253, 237, 185, 218, 94, 21, 70, 87, 167, 141, 157,
           132,
           144, 216, 171, 0, 140, 188, 211, 10, 247, 228, 88, 5, 184, 179, 69, 6, 208, 44, 30, 143, 202, 63, 15, 2, 193,
           175,
           189, 3, 1, 19, 138, 107, 58, 145, 17, 65, 79, 103, 220, 234, 151, 242, 207, 206, 240, 180, 230, 115, 150,
           172,
           116, 34, 231, 173, 53, 133, 226, 249, 55, 232, 28, 117, 223, 110, 71, 241, 26, 113, 29, 41, 197, 137, 111,
           183, 98,
           14, 170, 24, 190, 27, 252, 86, 62, 75, 198, 210, 121, 32, 154, 219, 192, 254, 120, 205, 90, 244, 31, 221,
           168, 51,
           136, 7, 199, 49, 177, 18, 16, 89, 39, 128, 236, 95, 96, 81, 127, 169, 25, 181, 74, 13, 45, 229, 122, 159,
           147, 201,
           156, 239, 160, 224, 59, 77, 174, 42, 245, 176, 200, 235, 187, 60, 131, 83, 153, 97, 23, 43, 4, 126, 186, 119,
           214,
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
# 3) ✅ LMMD Loss（修正版）
# =========================
class LMMD_loss(nn.Module):
    """
    Local Maximum Mean Discrepancy (LMMD)

    参考论文：Deep Subdomain Adaptation Network (DSAN)
    """

    def __init__(self, class_num=256, kernel_mul=2.0, kernel_num=5, fix_sigma=None,
                 use_confidence_filter=False, confidence_threshold=0.3):
        super(LMMD_loss, self).__init__()
        self.class_num = class_num
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        self.use_confidence_filter = use_confidence_filter
        self.confidence_threshold = confidence_threshold

    @staticmethod
    def _onehot(labels: np.ndarray, class_num: int):
        """将标签转换为 one-hot 编码"""
        return np.eye(class_num, dtype=np.float32)[labels]

    def guassian_kernel(self, source, target):
        """计算高斯核矩阵"""
        n_samples = int(source.size(0)) + int(target.size(0))
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
        total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
        L2_distance = ((total0 - total1) ** 2).sum(2)

        if self.fix_sigma is not None:
            bandwidth = self.fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.detach()) / (n_samples ** 2 - n_samples)

        bandwidth = bandwidth / (self.kernel_mul ** (self.kernel_num // 2))
        bandwidth_list = [bandwidth * (self.kernel_mul ** i) for i in range(self.kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bw) for bw in bandwidth_list]

        return sum(kernel_val)

    def cal_weight(self, s_label, t_label, batch_size: int):
        """计算 LMMD 权重矩阵"""
        s_label_np = s_label.detach().cpu().numpy().astype(np.int64)
        s_vec = self._onehot(s_label_np, self.class_num)

        s_sum = np.sum(s_vec, axis=0, keepdims=True)
        s_sum = s_sum + 1e-6
        s_vec = s_vec / s_sum

        t_prob = t_label.detach().cpu().numpy().astype(np.float32)
        t_hard = np.argmax(t_prob, axis=1).astype(np.int64)

        t_sum = np.sum(t_prob, axis=0, keepdims=True)
        t_sum = t_sum + 1e-6
        t_vec = t_prob / t_sum

        if self.use_confidence_filter:
            t_max_prob = np.max(t_prob, axis=1)
            confident_mask = t_max_prob > self.confidence_threshold

            if np.sum(confident_mask) > 0:
                t_hard_filtered = t_hard[confident_mask]
                index = list(set(s_label_np.tolist()) & set(t_hard_filtered.tolist()))
            else:
                index = list(set(s_label_np.tolist()) & set(t_hard.tolist()))
        else:
            index = list(set(s_label_np.tolist()) & set(t_hard.tolist()))

        mask = np.zeros((batch_size, self.class_num), dtype=np.float32)
        if len(index) > 0:
            mask[:, index] = 1.0

        s_vec = s_vec * mask
        t_vec = t_vec * mask

        weight_ss = np.matmul(s_vec, s_vec.T)
        weight_tt = np.matmul(t_vec, t_vec.T)
        weight_st = np.matmul(s_vec, t_vec.T)

        length = len(index)
        if length > 0:
            weight_ss = weight_ss / float(length)
            weight_tt = weight_tt / float(length)
            weight_st = weight_st / float(length)
        else:
            weight_ss = np.zeros((batch_size, batch_size), dtype=np.float32)
            weight_tt = np.zeros((batch_size, batch_size), dtype=np.float32)
            weight_st = np.zeros((batch_size, batch_size), dtype=np.float32)

        return weight_ss, weight_tt, weight_st

    def get_loss(self, source, target, s_label, t_prob):
        """计算 LMMD 损失"""
        batch_size = int(source.size(0))
        weight_ss, weight_tt, weight_st = self.cal_weight(s_label, t_prob, batch_size)

        device = source.device
        weight_ss = torch.from_numpy(weight_ss).to(device)
        weight_tt = torch.from_numpy(weight_tt).to(device)
        weight_st = torch.from_numpy(weight_st).to(device)

        kernels = self.guassian_kernel(source, target)

        SS = kernels[:batch_size, :batch_size]
        TT = kernels[batch_size:, batch_size:]
        ST = kernels[:batch_size, batch_size:]

        loss = torch.sum(weight_ss * SS + weight_tt * TT - 2.0 * weight_st * ST)

        # ✅ 修正：不指定 requires_grad
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.zeros((), device=device)

        return loss


# =========================
# 4) Network
# =========================
class CDP_Net(nn.Module):
    def __init__(self, num_classes=256):
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

        self.lmmd_loss = LMMD_loss(
            class_num=num_classes,
            use_confidence_filter=True,
            confidence_threshold=0.3
        )

    def forward(self, source, target, s_label):
        """
        训练模式：返回源域预测 + LMMD损失
        """
        s_feat = self.features(source)
        s0 = s_feat.view(s_feat.size(0), -1)
        s1 = self.classifier_1(s0)
        s_logits = self.final_classifier(s1)

        t_feat = self.features(target)
        t0 = t_feat.view(t_feat.size(0), -1)
        t1 = self.classifier_1(t0)
        t_logits = self.final_classifier(t1)

        t_prob = torch.softmax(t_logits.detach(), dim=1)

        lmmd = self.lmmd_loss.get_loss(s0, t0, s_label, t_prob)
        lmmd = lmmd + self.lmmd_loss.get_loss(s1, t1, s_label, t_prob)

        return s_logits, lmmd


# =========================
# 5) Logit Adjustment
# =========================
def compute_logit_adjustment(y_labels, tro=1.0, classes=9, eps=1e-12):
    """Logit Adjustment for Long-Tailed Recognition"""
    counts = np.bincount(y_labels.astype(np.int64), minlength=classes).astype(np.float64)
    pi = counts / max(counts.sum(), 1.0)
    adj = np.log(np.power(pi, tro) + eps).astype(np.float32)
    return adj


# =========================
# 6) Train / Validation
# =========================
def CDP_train(epoch, model):
    """训练一个 epoch"""
    model.train()
    clf_criterion = nn.CrossEntropyLoss()

    iter_source = iter(source_train_loader)
    iter_target = iter(target_finetune_loader)
    num_iter_target = len(target_finetune_loader)

    # 预取 target batch（CPU）
    finetune_trace_all = []
    for _ in range(num_iter_target):
        data_batch, _ = next(iter_target)
        finetune_trace_all.append(data_batch)

    num_iter = len(source_train_loader)

    for i in range(1, num_iter + 1):
        source_data, source_label = next(iter_source)

        idx = (i - 1) % num_iter_target
        target_data = finetune_trace_all[idx]

        source_data = source_data.to(device, non_blocking=True)
        source_label = source_label.to(device, non_blocking=True)
        target_data = target_data.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # LMMD forward
        logits, lmmd_loss_val = model(source_data, target_data, source_label)

        # 训练阶段：可选 logit adjustment
        if adjust_flag:
            logits = logits + adjustments

        clf_loss = clf_criterion(logits, source_label)

        # 总损失：Ltotal = alpha * Lcls + (1-alpha) * LLMMD
        loss = clf_loss + lambda_ * lmmd_loss_val

        loss.backward()
        optimizer.step()

        if i % log_interval == 0:
            print(
                'Train Epoch {}: [{}/{} ({:.0f}%)]\t'
                'total_loss: {:.6f}\tclf_loss: {:.6f}\t'
                'lmmd_loss: {:.6f}\talpha: {:.3f}'.format(
                    epoch,
                    i * len(source_data),
                    len(source_train_loader.dataset),
                    100. * i / len(source_train_loader),
                    loss.item(),
                    clf_loss.item(),
                    lmmd_loss_val.item(),
                    lambda_
                )
            )


def CDP_validation(model):
    """
    ✅ 验证阶段：不使用 logit adjustment
    """
    clf_criterion = nn.CrossEntropyLoss()
    model.eval()

    iter_source = iter(source_valid_loader)
    iter_target = iter(target_finetune_loader)
    num_iter_target = len(target_finetune_loader)

    finetune_trace_all = []
    for _ in range(num_iter_target):
        data_batch, _ = next(iter_target)
        finetune_trace_all.append(data_batch)

    num_iter = len(source_valid_loader)

    total_clf_loss = 0.0
    total_lmmd_loss = 0.0
    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for i in range(1, num_iter + 1):
            source_data, source_label = next(iter_source)
            target_data = finetune_trace_all[(i - 1) % num_iter_target]

            source_data = source_data.to(device, non_blocking=True)
            source_label = source_label.to(device, non_blocking=True)
            target_data = target_data.to(device, non_blocking=True)

            # LMMD forward
            logits, lmmd_loss_val = model(source_data, target_data, source_label)

            # ✅ 验证不加 adjustments
            clf_loss = clf_criterion(logits, source_label)

            loss = clf_loss + lambda_ * lmmd_loss_val

            total_clf_loss += clf_loss.item()
            total_lmmd_loss += lmmd_loss_val.item()
            total_loss += loss.item()

            pred = logits.data.max(1)[1]
            correct += pred.eq(source_label.data.view_as(pred)).sum().item()

    n = len(source_valid_loader)
    total_loss /= n
    total_clf_loss /= n
    total_lmmd_loss /= n

    acc = 100.0 * correct / len(source_valid_loader.dataset)
    print(
        'Validation: total_loss: {:.4f}, clf_loss: {:.4f}, '
        'lmmd_loss: {:.4f}, accuracy: {}/{} ({:.2f}%)'.format(
            total_loss, total_clf_loss, total_lmmd_loss,
            correct, len(source_valid_loader.dataset), acc
        )
    )
    return total_loss

# =========================
# 7) Main
# =========================
if __name__ == '__main__':
    # ===== 超参数配置 =====
    source_device_id = 1
    target_device_id = 2
    real_key_01 = 0x21
    real_key_02 = 0xCD

    labeling_method = 'identity'

    batch_size = 200
    total_epoch = 100
    finetune_epoch = 15
    lr = 0.001
    log_interval = 50

    train_num = 85000
    valid_num = 5000
    source_test_num = 9900
    target_finetune_num = 200
    target_test_num = 9400

    trace_offset = 0
    trace_length = 1000

    lambda_ = 0.05

    source_file_path = './Data/device1/'
    target_file_path = './Data/device2/'

    no_cuda = False
    cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    seed = 8
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    # ===== 加载数据 =====
    X_train_source = np.load(source_file_path + 'X_train.npy')
    Y_train_source = np.load(source_file_path + 'Y_train.npy')
    X_attack_target = np.load(target_file_path + 'X_attack.npy')
    Y_attack_target = np.load(target_file_path + 'Y_attack.npy')

    if labeling_method == 'identity':
        class_num = 256
    else:
        class_num = 9
        Y_train_source = calculate_HW(Y_train_source)
        Y_attack_target = calculate_HW(Y_attack_target)

    # ===== 数据标准化 =====
    mn = np.repeat(np.mean(X_train_source, axis=1, keepdims=True), X_train_source.shape[1], axis=1)
    std = np.repeat(np.std(X_train_source, axis=1, keepdims=True), X_train_source.shape[1], axis=1)
    X_train_source = (X_train_source - mn) / (std + 1e-12)

    mn = np.repeat(np.mean(X_attack_target, axis=1, keepdims=True), X_attack_target.shape[1], axis=1)
    std = np.repeat(np.std(X_attack_target, axis=1, keepdims=True), X_attack_target.shape[1], axis=1)
    X_attack_target = (X_attack_target - mn) / (std + 1e-12)

    # ===== 构建 DataLoader =====
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

    # ===== Logit Adjustment =====
    adjust_flag = False
    adj_np = compute_logit_adjustment(Y_train_source[0:train_num], tro=1.0, classes=class_num)
    adjustments = torch.from_numpy(adj_np).view(1, -1)
    if cuda:
        adjustments = adjustments.cuda(non_blocking=True)

    # ===== 构建模型 =====
    model = CDP_Net(num_classes=class_num)

    flag = "real" if adjust_flag else "fake"
    path = './models/ID' + str(flag) + '_pre-trained_cpda_device{}.pth'.format(source_device_id)
    print("Loading model:", path)

    checkpoint = None
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location='cuda' if cuda else 'cpu')
        model_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(model_dict, strict=False)
        print(f"✅ Model loaded from {path}")
    else:
        print(f"⚠️ Warning: Pretrained model not found at {path}, training from scratch")

    optimizer = optim.Adam([
        {'params': model.features.parameters()},
        {'params': model.classifier_1.parameters()},
        {'params': model.final_classifier.parameters()},
    ], lr=lr)

    if cuda:
        model.cuda()


    min_loss = 1e18

    if checkpoint is not None and isinstance(checkpoint, dict) and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("✅ Optimizer state loaded")

    # ===== 训练循环 =====
    print("\n" + "=" * 60)
    print("Starting Fine-tuning with LMMD...")
    print("=" * 60 + "\n")
    start_time = time.time()
    end_time = time.time()
    for epoch in range(1, finetune_epoch + 1):
        print(f'\n📌 Train Epoch {epoch}:')
        CDP_train(epoch, model)

        with torch.no_grad():
            valid_loss = CDP_validation(model)

            if valid_loss < min_loss:
                min_loss = valid_loss
                if not os.path.exists('./models'):
                    os.makedirs('./models')

                save_path = './models/IDlmmd' + str(flag) + '_best_valid_loss_fine_tuned_cpda_device{}_to_{}.pth'.format(
                    source_device_id, target_device_id
                )
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'min_loss': min_loss
                }, save_path)
                print(f'✅ Best model saved at epoch {epoch} with validation loss: {valid_loss:.4f}')

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n总训练时间: {elapsed_time:.2f} 秒")
    # 如果需要更友好的格式，可以转换为分钟和秒
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60
    print(f"总训练时间: {minutes} 分 {seconds:.2f} 秒")
