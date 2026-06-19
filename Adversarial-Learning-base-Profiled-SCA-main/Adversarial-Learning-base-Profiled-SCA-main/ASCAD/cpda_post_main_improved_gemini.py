import os
from torch.utils.data import Dataset, DataLoader
import torch
from torch import optim
import numpy as np
from torch import nn
import torch.nn.functional as F
import random

# 【优化】开启 cudnn 自动寻优
torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


### handle the dataset
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
        # 【优化】直接转换 Tensor
        trace_tensor = torch.from_numpy(trace).float().unsqueeze(0)
        label_tensor = torch.tensor(self.label_file[index], dtype=torch.long)
        return trace_tensor, label_tensor

    def __len__(self):
        return self.trace_num


### data loader for training
def load_training(batch_size, kwargs):
    data = TorchDataset(**kwargs)
    # 【优化】Windows下建议 num_workers=0 避免多进程报错，Linux可设为4
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True,
                                               drop_last=True, num_workers=0,
                                               pin_memory=True)
    return train_loader


### data loader for testing
def load_testing(batch_size, kwargs):
    data = TorchDataset(**kwargs)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False,
                                              drop_last=True, num_workers=0,
                                              pin_memory=True)
    return test_loader


# Sbox and HW tables
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

HW_byte_np = np.array(HW_byte, dtype=np.int32)


# --- 辅助函数：计算 One-Hot ---
def get_one_hot(labels, num_classes):
    batch_size = labels.size(0)
    one_hot = torch.zeros(batch_size, num_classes).to(labels.device)
    one_hot.scatter_(1, labels.unsqueeze(1), 1)
    return one_hot


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


# --- DCAN 核心：CMMD Loss ---
class CMMD_Loss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(CMMD_Loss, self).__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.lambda_reg = 0.1  # 正则化参数 lambda，防止矩阵不可逆

    def forward(self, source_feature, source_label, target_feature, target_label, num_classes):
        # 1. 计算 One-Hot 标签矩阵
        phi_s = get_one_hot(source_label, num_classes)
        phi_t = get_one_hot(target_label, num_classes)

        # 2. 计算标签核矩阵 L = Phi * Phi^T
        L_s = torch.mm(phi_s, phi_s.t())
        L_t = torch.mm(phi_t, phi_t.t())

        # 3. 计算正则化逆矩阵 (L + lambda * I)^-1
        batch_size_s = source_feature.size(0)
        batch_size_t = target_feature.size(0)

        L_s_tilde_inv = torch.inverse(L_s + self.lambda_reg * torch.eye(batch_size_s).to(source_feature.device))
        L_t_tilde_inv = torch.inverse(L_t + self.lambda_reg * torch.eye(batch_size_t).to(target_feature.device))

        # 4. 计算矩阵 G
        G_s = torch.mm(torch.mm(L_s_tilde_inv, L_s), L_s_tilde_inv)
        G_t = torch.mm(torch.mm(L_t_tilde_inv, L_t), L_t_tilde_inv)
        G_ts = torch.mm(torch.mm(L_t_tilde_inv, phi_t), torch.mm(phi_s.t(), L_s_tilde_inv))

        # 5. 计算特征的高斯核矩阵 K
        kernels = guassian_kernel(source_feature, target_feature,
                                  kernel_mul=self.kernel_mul, kernel_num=self.kernel_num)

        K_s = kernels[:batch_size_s, :batch_size_s]
        K_t = kernels[batch_size_s:, batch_size_s:]
        K_st = kernels[:batch_size_s, batch_size_s:]  # Source(rows) -> Target(cols) shape: (ns, nt)

        # 6. 计算最终 CMMD Loss
        # 修正: G_ts (nt, ns) * K_st (ns, nt) -> (nt, nt) 矩阵，可以求迹
        # 去掉了 G_ts.t()，因为 G_ts 和 K_st 的维度已经匹配可以相乘了
        loss = torch.trace(torch.mm(G_s, K_s)) + \
               torch.trace(torch.mm(G_t, K_t)) - \
               2 * torch.trace(torch.mm(G_ts, K_st))
        return loss


# --- DCAN 核心：互信息 Loss ---
class MI_Loss(nn.Module):
    def forward(self, target_probs):
        # 1. 条件熵 H(Y|X) = - sum(p * log p)
        conditional_entropy = -torch.mean(torch.sum(target_probs * torch.log(target_probs + 1e-5), dim=1))
        # 2. 边缘分布 P(Y_hat) = mean(p)
        marginal_prob = torch.mean(target_probs, dim=0)
        marginal_entropy = -torch.sum(marginal_prob * torch.log(marginal_prob + 1e-5))
        # Loss = 条件熵 - 边缘熵 (即 -MI)
        return conditional_entropy - marginal_entropy


# --- 改进后的模型：集成 CMMD 和 MI ---
class CDP_Net(nn.Module):
    def __init__(self, num_classes=9):
        super(CDP_Net, self).__init__()
        self.num_classes = num_classes
        # Encoder part
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=1), nn.SELU(), nn.BatchNorm1d(32), nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=50), nn.SELU(), nn.BatchNorm1d(64), nn.AvgPool1d(kernel_size=50, stride=50),
            nn.Conv1d(64, 128, kernel_size=3), nn.SELU(), nn.BatchNorm1d(128), nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        self.classifier_1 = nn.Sequential(nn.Linear(256, 20), nn.SELU())
        self.classifier_2 = nn.Sequential(nn.Linear(20, 20), nn.SELU())
        self.classifier_3 = nn.Sequential(nn.Linear(20, 20), nn.SELU())
        self.final_classifier = nn.Sequential(nn.Linear(20, num_classes))

        # DCAN Losses
        self.cmmd_loss_func = CMMD_Loss()
        self.mi_loss_func = MI_Loss()

        # Hyper-parameters
        self.gamma_0 = 0.95  # 伪标签阈值
        self.lambda_0 = 0.1  # CMMD 权重
        self.lambda_1 = 0.2  # MI 权重

    def forward(self, source, target, source_label=None):
        # Source flow
        source_f = self.features(source)
        source_0 = source_f.view(source_f.size(0), -1)
        source_1 = self.classifier_1(source_0)
        source_2 = self.classifier_2(source_1)
        source_3 = self.classifier_3(source_2)
        source_preds = self.final_classifier(source_3)

        if target is None:
            return source_preds, 0.0

        # Target flow
        target_f = self.features(target)
        target_0 = target_f.view(target_f.size(0), -1)  # Feature Z for CMMD
        target_1 = self.classifier_1(target_0)
        target_2 = self.classifier_2(target_1)
        target_3 = self.classifier_3(target_2)
        target_logits = self.final_classifier(target_3)
        target_probs = F.softmax(target_logits, dim=1)

        domain_loss = torch.tensor(0.0).to(source.device)

        # 计算 Loss (仅当提供了 source_label 时)
        if source_label is not None:
            # 1. 伪标签生成 (Pseudo-label generation)
            max_probs, pseudo_labels = torch.max(target_probs, dim=1)
            # 筛选高置信度样本
            mask = max_probs.ge(self.gamma_0)

            if torch.sum(mask) > 0:
                masked_target_z = target_0[mask]
                masked_pseudo_labels = pseudo_labels[mask]

                # CMMD Loss
                cmmd_val = self.cmmd_loss_func(source_0, source_label,
                                               masked_target_z, masked_pseudo_labels,
                                               self.num_classes)
            else:
                cmmd_val = torch.tensor(0.0).to(source.device)

            # 2. MI Loss
            mi_val = self.mi_loss_func(target_probs)

            # Weighted Sum
            domain_loss = self.lambda_0 * cmmd_val + self.lambda_1 * mi_val

        return source_preds, domain_loss


def compute_adjustment_1(Y_profiling, tro, classes=9, eps=1e-12):
    counts = np.bincount(Y_profiling, minlength=classes).astype(np.float64)
    pi = counts / np.maximum(counts.sum(), 1.0)
    adjustments = np.log(np.power(pi, tro) + eps)
    return adjustments.astype(np.float32)


def calculate_HW(data):
    return HW_byte_np[data.astype(int)]


def CDP_train(epoch, model):
    model.train()  # 确保开启训练模式

    iter_source = iter(source_train_loader)
    iter_target = iter(target_finetune_loader)
    num_iter_target = len(target_finetune_loader)

    finetune_trace_all = torch.zeros((num_iter_target, batch_size, 1, trace_length))
    for i in range(num_iter_target):
        try:
            data_batch, _ = next(iter_target)
        except StopIteration:
            iter_target = iter(target_finetune_loader)
            data_batch, _ = next(iter_target)
        finetune_trace_all[i, :, :, :] = data_batch

    num_iter = len(source_train_loader)
    clf_criterion = nn.CrossEntropyLoss()

    for i in range(1, num_iter + 1):
        try:
            source_data, source_label = next(iter_source)
        except StopIteration:
            iter_source = iter(source_train_loader)
            source_data, source_label = next(iter_source)

        target_data = finetune_trace_all[(i - 1) % num_iter_target, :, :, :]

        if cuda:
            source_data = source_data.cuda(non_blocking=True)
            source_label = source_label.cuda(non_blocking=True)
            target_data = target_data.cuda(non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # 传入 source_label，计算 DCAN loss (CMMD + MI)
        source_preds, domain_loss = model(source_data, target_data, source_label)

        if adjust_flag:
            source_preds = source_preds + 1 * adjustments

        clf_loss = clf_criterion(source_preds, source_label)

        # 总损失：分类损失 + 域适配损失
        loss = clf_loss + domain_loss

        loss.backward()
        optimizer.step()

        if i % log_interval == 0:
            print('Train Epoch {}: [{}/{} ({:.0f}%)]\ttotal_loss: {:.6f}\tclf_loss: {:.6f}\tdomain_loss: {:.6f}'.format(
                epoch, i * len(source_data), len(source_train_loader) * batch_size,
                       100. * i / len(source_train_loader), loss.item(), clf_loss.item(), domain_loss.item()))


def CDP_validation(model):
    clf_criterion = nn.CrossEntropyLoss()
    model.eval()

    iter_source = iter(source_valid_loader)
    iter_target = iter(target_finetune_loader)
    num_iter_target = len(target_finetune_loader)

    finetune_trace_all = torch.zeros((num_iter_target, batch_size, 1, trace_length))
    for i in range(num_iter_target):
        try:
            data_batch, _ = next(iter_target)
        except StopIteration:
            iter_target = iter(target_finetune_loader)
            data_batch, _ = next(iter_target)
        finetune_trace_all[i, :, :, :] = data_batch

    num_iter = len(source_valid_loader)
    total_clf_loss = 0
    total_domain_loss = 0
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for i in range(1, num_iter + 1):
            try:
                source_data, source_label = next(iter_source)
            except StopIteration:
                iter_source = iter(source_valid_loader)
                source_data, source_label = next(iter_source)

            target_data = finetune_trace_all[(i - 1) % num_iter_target, :, :, :]

            if cuda:
                source_data = source_data.cuda(non_blocking=True)
                source_label = source_label.cuda(non_blocking=True)
                target_data = target_data.cuda(non_blocking=True)

            # 计算验证集的 Loss
            valid_preds, domain_loss = model(source_data, target_data, source_label)
            clf_loss = clf_criterion(valid_preds, source_label)

            loss = clf_loss + domain_loss

            total_clf_loss += clf_loss.item()
            total_domain_loss += domain_loss.item()
            total_loss += loss.item()

            pred = valid_preds.data.max(1)[1]
            correct += pred.eq(source_label.data.view_as(pred)).cpu().sum()

    total_loss /= len(source_valid_loader)
    total_clf_loss /= len(source_valid_loader)
    total_domain_loss /= len(source_valid_loader)

    print('Validation: total_loss: {:.4f}, clf_loss: {:.4f}, domain_loss: {:.4f}, accuracy: {}/{} ({:.2f}%)'.format(
        total_loss, total_clf_loss, total_domain_loss, correct, len(source_valid_loader.dataset),
        100. * correct / len(source_valid_loader.dataset)))

    return total_loss


def addClockJitter(traces, clock_range, trace_length):
    print('Add clock jitters...')
    output_traces = []
    # min_trace_length = 100000 # Unused
    for trace_idx in range(len(traces)):
        if (trace_idx % 2000 == 0):
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


if __name__ == '__main__':
    source_device_id = 0
    target_device_id = 1
    real_key_01 = 224
    real_key_02 = 224
    labeling_method = 'hw'
    # lambda_ = 0.1 # 原 lambda 不再使用，已被 DCAN 内部参数 lambda_0, lambda_1 替代
    preprocess = None
    batch_size = 200
    total_epoch = 100
    finetune_epoch = 25
    lr = 0.001
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
    adjust_flag = 0  # 确保变量已定义

    no_cuda = False
    cuda = not no_cuda and torch.cuda.is_available()
    seed = 8
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    # 模拟加载数据 (确保路径存在或替换为真实数据加载)
    try:
        X_train_source = np.load(source_file_path + 'X_train.npy')
        Y_train_source = np.load(source_file_path + 'Y_train.npy')
        X_attack_source = np.load(source_file_path + 'X_attack.npy')
        Y_attack_source = np.load(source_file_path + 'Y_attack.npy')
    except FileNotFoundError:
        print("Error: Data files not found. Please check 'source_file_path'.")
        # 为演示代码运行，这里使用随机数据填充 (实际运行时请移除)
        X_train_source = np.random.randn(50000, 1000).astype(np.float32)
        Y_train_source = np.random.randint(0, 256, 50000).astype(np.uint8)
        X_attack_source = np.random.randn(10000, 1000).astype(np.float32)
        Y_attack_source = np.random.randint(0, 256, 10000).astype(np.uint8)

    # add clock jitter to the target domain
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

    if adjust_flag:
        adjustments = compute_adjustment_1(Y_train_source[0:train_num], 1.0)
        if cuda:
            adjustments = torch.from_numpy(adjustments).cuda()

    print('Load data complete!')

    model = CDP_Net(num_classes=9)
    pretrained_path = (
                './models/True' + '_' + str(countermeasure) + '_pre-trained_cpda_device{}.pth'.format(
            source_device_id))
    print("Loading model:", pretrained_path)

    # 加载预训练模型
    if os.path.exists(pretrained_path):
        checkpoint = torch.load(pretrained_path)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 兼容可能直接保存 model 的情况
            model.load_state_dict(checkpoint)
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

    # 加载 Optimizer 状态
    if os.path.exists(pretrained_path) and isinstance(checkpoint, dict) and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

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
                }, './models/gemini'+ str(
                    countermeasure) + '_best_valid_loss_fine_tuned_cpda_2_device{}_to_{}.pth'.format(source_device_id,
                                                                                                   target_device_id))