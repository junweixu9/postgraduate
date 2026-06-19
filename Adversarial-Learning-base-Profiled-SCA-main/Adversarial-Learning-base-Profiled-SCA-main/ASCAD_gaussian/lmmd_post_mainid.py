import os
import time

from torch.utils.data import Dataset, DataLoader
import torch
from torch import optim
import numpy as np
from torch import nn
import torch.nn.functional as F

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
# 2) HW Table
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


# =========================
# 3) LMMD Loss
# =========================
class LMMD_loss(nn.Module):
    """
    Local Maximum Mean Discrepancy (LMMD)
    参考论文：Deep Subdomain Adaptation Network (DSAN)

    本版本修复了以下三个问题：
      [修复1] NaN/Inf 检查：移至 kernel 计算后立即检查，避免梯度图已被污染；
              使用 .item() 规范 scalar tensor 的布尔判断。
      [修复2] confidence_filter 分支：过滤后的 t_vec 与 mask 数据源保持一致，
              仅对置信度足够高的目标样本参与权重计算，避免不一致导致的错误对齐。
      [修复3] guassian_kernel：对输入特征做 L2 归一化，消除多层特征维度差异
              （D=256 vs D=20）导致的 bandwidth 量级失衡，确保三层 LMMD 损失
              贡献量级一致。
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
        """
        计算多核高斯核矩阵，返回 [2B, 2B]。

        [修复3] 在计算 L2 距离前，对 source 和 target 特征做 L2 归一化。
        原因：三层特征维度分别为 D=256、D=20、D=20。
        高维空间 L2 距离期望值正比于 D，导致 D=256 层的 bandwidth 约是
        D=20 层的 12 倍以上，使高维层 kernel 矩阵退化为接近全 1 的常数矩阵，
        该层 LMMD 损失实质为零。
        L2 归一化将所有特征映射到单位超球面，L2 距离值域统一为 [0, 2]，
        bandwidth 估计在不同层间保持一致，三层损失贡献量级均衡。
        L2 归一化不改变特征方向（类别判别性），无副作用。
        """
        # ---- 修复3：L2 归一化统一各层特征尺度 ----
        source = F.normalize(source, p=2, dim=1)
        target = F.normalize(target, p=2, dim=1)
        # ------------------------------------------

        n_samples = int(source.size(0)) + int(target.size(0))
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
        total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
        L2_distance = ((total0 - total1) ** 2).sum(2)

        if self.fix_sigma is not None:
            bandwidth = self.fix_sigma
        else:
            # 归一化后 L2_distance ∈ [0, 2]，bandwidth 估计稳定
            bandwidth = torch.sum(L2_distance.detach()) / (n_samples ** 2 - n_samples + 1e-12)

        bandwidth = bandwidth / (self.kernel_mul ** (self.kernel_num // 2))
        bandwidth_list = [bandwidth * (self.kernel_mul ** i) for i in range(self.kernel_num)]
        kernel_val = [torch.exp(-L2_distance / (bw + 1e-12)) for bw in bandwidth_list]

        return sum(kernel_val)

    def cal_weight(self, s_label, t_label, batch_size: int):
        """
        计算 LMMD 权重矩阵。

        [修复2] confidence_filter 分支修复：
        原代码问题：当 use_confidence_filter=True 时，t_vec（用于权重计算）
        仍基于全部 t_prob（含低置信度样本），但 mask 只保留过滤后出现的类别，
        导致权重计算的数据源与类别筛选不一致——低置信度样本的 soft prob 贡献
        了权重矩阵，却没有通过 mask 被筛除，产生错误的对齐信号。

        修复方案：当启用 confidence_filter 时，对 t_vec 也只保留高置信度样本
        的 soft prob，低置信度样本对应行置零，再做列归一化和 mask 操作，
        确保权重计算与类别筛选完全基于同一子集数据。
        """
        s_label_np = s_label.detach().cpu().numpy().astype(np.int64)
        s_vec = self._onehot(s_label_np, self.class_num)   # [B, C] one-hot

        s_sum = np.sum(s_vec, axis=0, keepdims=True) + 1e-6
        s_vec = s_vec / s_sum                              # [B, C] 列归一化

        t_prob = t_label.detach().cpu().numpy().astype(np.float32)  # [B, C] soft
        t_hard = np.argmax(t_prob, axis=1).astype(np.int64)         # [B]

        if self.use_confidence_filter:
            # ---- 修复2：t_vec 与 mask 数据源保持一致 ----
            t_max_prob = np.max(t_prob, axis=1)                      # [B]
            confident_mask = t_max_prob > self.confidence_threshold  # [B] bool

            # 仅保留高置信度样本的 soft prob，低置信度行置零
            t_prob_filtered = t_prob.copy()
            t_prob_filtered[~confident_mask] = 0.0                   # [B, C]

            # 列归一化时仅计入高置信度样本的贡献
            t_sum = np.sum(t_prob_filtered, axis=0, keepdims=True) + 1e-6
            t_vec = t_prob_filtered / t_sum                          # [B, C]

            # 类别 index：source 出现的类 ∩ 高置信度 target 预测出现的类
            if np.sum(confident_mask) > 0:
                t_hard_filtered = t_hard[confident_mask]
                index = list(set(s_label_np.tolist()) & set(t_hard_filtered.tolist()))
            else:
                # 无高置信度样本：index 为空，loss 将为零
                index = []
            # -----------------------------------------------
        else:
            t_sum = np.sum(t_prob, axis=0, keepdims=True) + 1e-6
            t_vec = t_prob / t_sum                                   # [B, C]
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
            weight_ss /= float(length)
            weight_tt /= float(length)
            weight_st /= float(length)
        else:
            weight_ss = np.zeros((batch_size, batch_size), dtype=np.float32)
            weight_tt = np.zeros((batch_size, batch_size), dtype=np.float32)
            weight_st = np.zeros((batch_size, batch_size), dtype=np.float32)

        return weight_ss, weight_tt, weight_st

    def get_loss(self, source, target, s_label, t_prob):
        """
        计算 LMMD 损失。

        [修复1] NaN/Inf 检查位置修复：
        原代码在 loss 计算完成后才做 NaN 检查，此时若 kernel 矩阵中已有 NaN，
        梯度图已被污染，返回零值无法阻止反向传播中的 NaN 传播。
        修复：在 kernel 计算完成后立即检查，若 kernel 矩阵含 NaN/Inf 则
        直接返回零损失，完全跳过后续的权重计算和 loss 计算，保护梯度图。
        同时使用 .item() 将 scalar tensor 转为 Python bool，规范布尔判断，
        避免潜在的 RuntimeError。
        """
        batch_size = int(source.size(0))
        device = source.device

        # 计算 kernel 矩阵
        kernels = self.guassian_kernel(source, target)

        # ---- 修复1：kernel 计算后立即检查 NaN/Inf ----
        # 使用 .item() 将 scalar tensor 转为 Python bool，规范布尔判断
        if torch.isnan(kernels).any().item() or torch.isinf(kernels).any().item():
            return torch.zeros((), device=device)
        # -----------------------------------------------

        weight_ss, weight_tt, weight_st = self.cal_weight(s_label, t_prob, batch_size)

        weight_ss = torch.from_numpy(weight_ss).to(device)
        weight_tt = torch.from_numpy(weight_tt).to(device)
        weight_st = torch.from_numpy(weight_st).to(device)

        SS = kernels[:batch_size, :batch_size]
        TT = kernels[batch_size:, batch_size:]
        ST = kernels[:batch_size, batch_size:]

        loss = torch.sum(weight_ss * SS + weight_tt * TT - 2.0 * weight_st * ST)

        return loss


# =========================
# 4) CDP_Net
# =========================
class CDP_Net(nn.Module):
    def __init__(self, num_classes=256):
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
        self.final_classifier = nn.Sequential(nn.Linear(20, num_classes))

        self.lmmd_loss = LMMD_loss(
            class_num=num_classes,
            use_confidence_filter=False,
            confidence_threshold=0.3
        )

    def forward(self, source, target, s_label):
        # source flow
        source = self.features(source)
        source_0 = source.view(source.size(0), -1)
        source_1 = self.classifier_1(source_0)
        source_2 = self.classifier_2(source_1)
        source_3 = self.classifier_3(source_2)

        # target flow（完整前向传播以获取伪标签）
        target = self.features(target)
        target_0 = target.view(target.size(0), -1)
        target_1 = self.classifier_1(target_0)
        target_2 = self.classifier_2(target_1)
        target_3 = self.classifier_3(target_2)
        target_logits = self.final_classifier(target_3)

        # 软伪标签（detach，不反传到分类器）
        t_prob = torch.softmax(target_logits.detach(), dim=1)

        # 多层 LMMD 对齐（修复3：kernel 内部已做 L2 归一化，三层损失量级一致）
        lmmd = self.lmmd_loss.get_loss(source_0, target_0, s_label, t_prob)
        lmmd += self.lmmd_loss.get_loss(source_1, target_1, s_label, t_prob)
        lmmd += self.lmmd_loss.get_loss(source_2, target_2, s_label, t_prob)

        result = self.final_classifier(source_3)
        return result, lmmd


# =========================
# 5) Helpers
# =========================
def compute_adjustment_1(Y_profiling, tro, classes=9, eps=1e-12):
    counts = np.bincount(Y_profiling, minlength=classes).astype(np.float64)
    pi = counts / np.maximum(counts.sum(), 1.0)
    adjustments = np.log(np.power(pi, tro) + eps)
    return adjustments.astype(np.float32)


def calculate_HW(data):
    return HW_byte_np[data.astype(int)]


# =========================
# 6) Train / Validation
# =========================
def CDP_train(epoch, model):
    model.train()
    clf_criterion = nn.CrossEntropyLoss()

    iter_source = iter(source_train_loader)
    iter_target = iter(target_finetune_loader)
    num_iter_target = len(target_finetune_loader)

    finetune_trace_all = []
    for _ in range(num_iter_target):
        data_batch, _ = next(iter_target)
        finetune_trace_all.append(data_batch)

    num_iter = len(source_train_loader)

    for i in range(1, num_iter + 1):
        source_data, source_label = next(iter_source)
        target_data = finetune_trace_all[(i - 1) % num_iter_target]

        source_data = source_data.to(device, non_blocking=True)
        source_label = source_label.to(device, non_blocking=True)
        target_data = target_data.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        source_preds, lmmd_loss = model(source_data, target_data, source_label)

        if adjust_flag:
            source_preds = source_preds + adjustments

        clf_loss = clf_criterion(source_preds, source_label)
        loss = clf_loss + lambda_ * lmmd_loss
        loss.backward()
        optimizer.step()

        if i % log_interval == 0:
            print('Train Epoch {}: [{}/{} ({:.0f}%)]\t'
                  'total_loss: {:.6f}\tclf_loss: {:.6f}\t'
                  'lmmd_loss: {:.6f}\tlambda: {:.4f}'.format(
                epoch, i * len(source_data), len(source_train_loader.dataset),
                100. * i / len(source_train_loader),
                loss.item(), clf_loss.item(), lmmd_loss.item(), lambda_))


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
    total_clf_loss = 0
    total_mmd_loss = 0
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for i in range(1, num_iter + 1):
            source_data, source_label = next(iter_source)
            target_data = finetune_trace_all[(i - 1) % num_iter_target, :, :, :]

            if cuda:
                source_data = source_data.cuda(non_blocking=True)
                source_label = source_label.cuda(non_blocking=True)
                target_data = target_data.cuda(non_blocking=True)

            valid_preds, lmmd_loss = model(source_data, target_data, source_label)
            clf_loss = clf_criterion(valid_preds, source_label)
            loss = clf_loss + lambda_ * lmmd_loss

            total_clf_loss += clf_loss.item()
            total_mmd_loss += lmmd_loss.item()
            total_loss += loss.item()

            pred = valid_preds.data.max(1)[1]
            correct += pred.eq(source_label.data.view_as(pred)).cpu().sum()

    total_loss /= len(source_valid_loader)
    total_clf_loss /= len(source_valid_loader)
    total_mmd_loss /= len(source_valid_loader)

    print('Validation: total_loss: {:.4f}, clf_loss: {:.4f}, '
          'lmmd_loss: {:.4f}, accuracy: {}/{} ({:.2f}%)'.format(
        total_loss, total_clf_loss, total_mmd_loss,
        correct, len(source_valid_loader.dataset),
        100. * correct / len(source_valid_loader.dataset)))

    return total_loss


# =========================
# 7) Data Augmentation
# =========================
def addGaussianNoise(traces, noise_level):
    print('Add Gaussian noise...')
    if noise_level == 0:
        return traces
    else:
        output_traces = np.zeros(np.shape(traces))
        print(np.shape(output_traces))
        for trace in range(len(traces)):
            if trace % 5000 == 0:
                print(str(trace) + '/' + str(len(traces)))
            profile_trace = traces[trace]
            noise = np.random.normal(0, noise_level, size=np.shape(profile_trace))
            output_traces[trace] = profile_trace + noise
        return output_traces


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
# 8) Main
# =========================
if __name__ == '__main__':
    source_device_id = 0
    target_device_id = 1
    real_key_01 = 224
    real_key_02 = 224
    labeling_method = 'id'
    lambda_ = 0.1
    preprocess = None
    batch_size = 200
    total_epoch = 100
    finetune_epoch = 15
    lr = 0.001
    log_interval = 50
    train_num = 45000
    valid_num = 5000
    source_test_num = 10000
    target_finetune_num = batch_size
    trace_offset = 0
    trace_length = 700
    countermeasure = '_GaussianNoise_level4'
    noise_level = 4
    source_file_path = './Data/ASCAD/'

    no_cuda = False
    cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    seed = 8
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    X_train_source = np.load(source_file_path + 'X_train.npy')
    Y_train_source = np.load(source_file_path + 'Y_train.npy')
    X_attack_source = np.load(source_file_path + 'X_attack.npy')
    Y_attack_source = np.load(source_file_path + 'Y_attack.npy')

    X_attack_target = addGaussianNoise(X_attack_source, noise_level)
    Y_attack_target = np.load(source_file_path + 'Y_attack.npy')

    if labeling_method == 'id':
        class_num = 256
    elif labeling_method == 'hw':
        class_num = 9
        Y_train_source = calculate_HW(Y_train_source)
        Y_attack_target = calculate_HW(Y_attack_target)

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

    adjustments_np = compute_adjustment_1(Y_train_source[0:train_num], tro=1, classes=256)
    adjustments = torch.from_numpy(adjustments_np).view(1, -1)
    if cuda:
        adjustments = adjustments.cuda()

    adjust_flag = False
    flag = "real" if adjust_flag else "fake"

    model = CDP_Net(num_classes=256)
    pretrained_path = './models/best_pretrained.pth'
    print("Loading model:", pretrained_path)

    checkpoint = None
    if os.path.exists(pretrained_path):
        checkpoint = torch.load(pretrained_path)
        model_dict = checkpoint['model_state_dict']
        model.load_state_dict(model_dict)
    else:
        print(f"Warning: Pretrained model not found at {pretrained_path}")

    optimizer = optim.Adam([
        {'params': model.features.parameters()},
        {'params': model.classifier_1.parameters()},
        {'params': model.classifier_2.parameters()},
        {'params': model.classifier_3.parameters()},
        {'params': model.final_classifier.parameters()}
    ], lr=lr)

    if cuda:
        model.cuda()

    min_loss = 1000

    if checkpoint is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_time = time.time()
    end_time = time.time()
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
                }, './models/lmmdnew' + str(countermeasure) + '_' + str(flag) +
                   str(target_finetune_num) +
                   '_best_valid_loss_fine_tuned_cpda_device{}_to_{}.pth'.format(
                       source_device_id, target_device_id))

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n总训练时间: {elapsed_time:.2f} 秒")
    # 如果需要更友好的格式，可以转换为分钟和秒
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60
    print(f"总训练时间: {minutes} 分 {seconds:.2f} 秒")

    del source_train_loader
    del source_valid_loader
    del target_finetune_loader

    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    import gc
    gc.collect()

    print("Cleanup complete, exiting...")