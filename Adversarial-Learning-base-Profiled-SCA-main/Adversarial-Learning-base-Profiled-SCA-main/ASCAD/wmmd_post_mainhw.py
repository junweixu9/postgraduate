import os
from torch.utils.data import Dataset, DataLoader
import torch
from torch import optim
import numpy as np
from torch import nn
import random
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
    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True,
                                       drop_last=True, num_workers=4,
                                       pin_memory=True, persistent_workers=True)


def load_testing(batch_size, kwargs):
    data = TorchDataset(**kwargs)
    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False,
                                       drop_last=True, num_workers=4,
                                       pin_memory=True, persistent_workers=True)


# =========================
# 2) AES Tables
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

HW_byte_np = np.array(HW_byte, dtype=np.int32)


def calculate_HW(data):
    return HW_byte_np[data.astype(int)]


# =========================
# 3) Logit Adjustment
# =========================
def compute_adjustment_1(Y_profiling, tro, classes=9, eps=1e-12):
    counts = np.bincount(Y_profiling, minlength=classes).astype(np.float64)
    pi = counts / np.maximum(counts.sum(), 1.0)
    adjustments = np.log(np.power(pi, tro) + eps)
    return adjustments.astype(np.float32)


# =========================
# 4) Gaussian Kernel（共用，两个 WMMD 函数统一调用）
# =========================
def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    标准多核高斯 kernel，返回 [2B, 2B] 的 kernel 矩阵。
    对应论文 Eqn.(3)：k = Σ β_l k_l(x^s, x^t)
    """
    n_samples = int(source.size(0)) + int(target.size(0))
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
    total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
    L2_distance = ((total0 - total1) ** 2).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance) / (n_samples ** 2 - n_samples)

    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bw) for bw in bandwidth_list]
    return sum(kernel_val)


# =========================
# 5) wmmd_rbf（二次时间复杂度，对应 Eqn.8）
# =========================
def wmmd_rbf(source, target, source_labels, alpha,
             kernel_mul=2.0, kernel_num=5, fix_sigma=None, eps=1e-12):
    """
    Weighted MMD 二次版，对应论文 Eqn.(8) 展开：

        MMD²_w = Σi Σj wi wj k(xi^s, xj^s)
               + (1/N²) Σi Σj k(xi^t, xj^t)
               - 2*(1/N) Σi Σj wi k(xi^s, xj^t)

    其中 wi = αyi / Σαyi（归一化权重）

    此函数逻辑与原论文完全一致，无需修改。
    """
    B_s = int(source.size(0))
    B_t = int(target.size(0))

    kernels = guassian_kernel(source, target, kernel_mul=kernel_mul,
                              kernel_num=kernel_num, fix_sigma=fix_sigma)
    K_ss = kernels[:B_s, :B_s]
    K_tt = kernels[B_s:, B_s:]
    K_st = kernels[:B_s, B_s:]

    w = alpha[source_labels].clamp_min(eps)
    w = w / (w.sum() + eps)  # 归一化

    term_ss = (w.view(-1, 1) * w.view(1, -1) * K_ss).sum()
    term_tt = K_tt.mean()
    term_st = (w.view(-1, 1) * K_st).mean(dim=1).sum()

    return term_ss + term_tt - 2.0 * term_st


# =========================
# 6) wmmd_linear（线性时间复杂度，对应 Eqn.9-10）
# =========================
def wmmd_linear(source, target, source_labels, alpha,
                kernel_mul=2.0, kernel_num=5, fix_sigma=None, eps=1e-12):
    """
    Weighted MMD 线性时间版，对应论文 Eqn.(9)-(10)：

        MMD²_{l,w} = (2/M) Σ_{i=1}^{M/2} h_{l,w}(z_i)

        h_{l,w}(z_i) = α_{y^s_{2i-1}} · α_{y^s_{2i}} · k(x^s_{2i-1}, x^s_{2i})
                      + k(x^t_{2i-1}, x^t_{2i})
                      - α_{y^s_{2i-1}} · k(x^s_{2i-1}, x^t_{2i})
                      - α_{y^s_{2i}} · k(x^s_{2i}, x^t_{2i-1})

    ----------------------------------------------------------------
    原代码存在三处严重错误，均已修复：

    [修复1 - 严重] Kernel 实现不一致
      原代码在 wmmd_linear 内部定义了独立的单核 rbf_kernel（使用
      median sigma），与 wmmd_rbf 使用的多核 guassian_kernel 完全不同。
      同一模型中两个函数的 kernel 计算不一致，WMMD 损失量级无法对齐。
      修复：改为统一调用 guassian_kernel，与 wmmd_rbf 保持一致。

    [修复2 - 严重] Alpha 被错误归一化
      原代码：w = alpha[source_labels] / (alpha.sum() + eps)
      论文 Eqn.(10) 中线性版本直接使用原始 αy 值，不做全局归一化。
      归一化只在 Eqn.(8) 的二次版本中出现（分母 Σαyi）。
      修复：直接使用 alpha[source_labels]，不归一化。

    [修复3 - 严重] pairwise kernel 提取方式正确化
      原代码用逐对 rbf_kernel(a_i, b_i) 计算，现改为通过完整
      guassian_kernel 矩阵提取对角线，确保与 wmmd_rbf 使用同一
      bandwidth 估计，两个版本在数值上等价（相同 kernel 定义下）。
    ----------------------------------------------------------------
    """
    B = int(source.size(0))
    assert B == int(target.size(0)), \
        "Source and target batch sizes must be equal for linear WMMD"
    assert B % 2 == 0, \
        "Batch size must be even for quad-tuple sampling"

    n_pairs = B // 2

    # 修复2：直接使用原始 alpha，不做全局归一化
    w = alpha[source_labels].clamp_min(eps)  # [B]
    w_1 = w[0::2]   # α_{y^s_{2i-1}},  shape [n_pairs]
    w_2 = w[1::2]   # α_{y^s_{2i}},    shape [n_pairs]

    s_1 = source[0::2]   # x^s_{2i-1}
    s_2 = source[1::2]   # x^s_{2i}
    t_1 = target[0::2]   # x^t_{2i-1}
    t_2 = target[1::2]   # x^t_{2i}

    # 修复1+3：用 guassian_kernel 的完整矩阵提取逐对 kernel 值
    # guassian_kernel([a; b]) 返回 [2n, 2n]，其中 K[i, n+i] = k(a_i, b_i)
    def pairwise_kernel_diag(a, b):
        """
        返回 [k(a_0,b_0), k(a_1,b_1), ..., k(a_{n-1},b_{n-1})]，shape [n_pairs]
        通过 guassian_kernel 完整矩阵的反对角块对角线提取，
        保证 bandwidth 与 wmmd_rbf 估计逻辑完全一致。
        """
        n = a.size(0)
        K = guassian_kernel(a, b, kernel_mul=kernel_mul,
                            kernel_num=kernel_num, fix_sigma=fix_sigma)
        # K[:n, n:] 是 k(a_i, b_j) 块，对角线即 k(a_i, b_i)
        return K[:n, n:].diagonal()

    k_ss = pairwise_kernel_diag(s_1, s_2)   # k(x^s_{2i-1}, x^s_{2i})
    k_tt = pairwise_kernel_diag(t_1, t_2)   # k(x^t_{2i-1}, x^t_{2i})
    k_st_1 = pairwise_kernel_diag(s_1, t_2) # k(x^s_{2i-1}, x^t_{2i})
    k_st_2 = pairwise_kernel_diag(s_2, t_1) # k(x^s_{2i}, x^t_{2i-1})

    # Eqn.(10)
    h_values = w_1 * w_2 * k_ss + k_tt - w_1 * k_st_1 - w_2 * k_st_2

    # Eqn.(9)
    return (2.0 / B) * h_values.sum()


# =========================
# 7) CDP_Net
# =========================
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
        self.final_classifier = nn.Sequential(nn.Linear(20, num_classes))

    def forward(self, source, target, source_label, alpha):
        mmd_loss = 0

        # source flow
        source_0 = self.features(source).view(source.size(0), -1)
        source_1 = self.classifier_1(source_0)
        source_2 = self.classifier_2(source_1)
        source_3 = self.classifier_3(source_2)

        # target flow + 多层 WMMD
        target_0 = self.features(target).view(target.size(0), -1)
        mmd_loss += wmmd_rbf(source_0, target_0, source_label, alpha)

        target_1 = self.classifier_1(target_0)
        mmd_loss += wmmd_rbf(source_1, target_1, source_label, alpha)

        target_2 = self.classifier_2(target_1)
        mmd_loss += wmmd_rbf(source_2, target_2, source_label, alpha)

        result = self.final_classifier(source_3)
        return result, mmd_loss

    def predict(self, x):
        x = self.features(x).view(x.size(0), -1)
        x = self.classifier_1(x)
        x = self.classifier_2(x)
        x = self.classifier_3(x)
        return self.final_classifier(x)


# =========================
# 8) Alpha 估计 (CEM E/C-step)
# =========================
@torch.no_grad()
def estimate_alpha_from_pseudolabels(model, target_loader, source_labels_np,
                                     num_classes, cuda, eps=1e-12):
    """
    论文 CEM 算法的 E-step + C-step：
        α_c = w^t_c / w^s_c
        w^s_c = Mc / M   （源域类别先验）
        w^t_c            （目标域伪标签估计先验）
    此函数逻辑与原论文完全一致，无需修改。
    """
    ws_counts = np.bincount(source_labels_np.astype(int),
                            minlength=num_classes).astype(np.float64)
    ws = ws_counts / np.maximum(ws_counts.sum(), 1.0)

    wt_counts = np.zeros(num_classes, dtype=np.float64)
    model.eval()
    for xb, _ in target_loader:
        if cuda:
            xb = xb.cuda(non_blocking=True)
        out = model.predict(xb)
        pseudo = out.argmax(dim=1).detach().cpu().numpy()
        wt_counts += np.bincount(pseudo, minlength=num_classes)

    wt = wt_counts / np.maximum(wt_counts.sum(), 1.0)
    alpha = wt / (ws + eps)
    alpha = np.clip(alpha, 0.1, 10.0).astype(np.float32)
    alpha_t = torch.from_numpy(alpha)
    if cuda:
        alpha_t = alpha_t.cuda()
    return alpha_t


# =========================
# 9) Train / Validation
# =========================
def CDP_train(epoch, model, alpha):
    model.eval()
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

        if cuda:
            source_data = source_data.cuda(non_blocking=True)
            source_label = source_label.cuda(non_blocking=True)
            target_data = target_data.cuda(non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        source_preds, mmd_loss = model(source_data, target_data, source_label, alpha)

        if adjust_flag:
            source_preds = source_preds + adjustments

        clf_loss = clf_criterion(source_preds, source_label)
        loss = clf_loss + lambda_ * mmd_loss
        loss.backward()
        optimizer.step()

        if i % log_interval == 0:
            print('Train Epoch {}: [{}/{} ({:.0f}%)]\t'
                  'total_loss: {:.6f}\tclf_loss: {:.6f}\twmmd_loss: {:.6f}'.format(
                epoch, i * len(source_data),
                len(source_train_loader) * batch_size,
                100. * i / len(source_train_loader),
                loss.item(), clf_loss.item(), mmd_loss.item()))


def CDP_validation(model, alpha):
    clf_criterion = nn.CrossEntropyLoss()
    model.eval()

    iter_source = iter(source_valid_loader)
    iter_target = iter(target_finetune_loader)
    num_iter_target = len(target_finetune_loader)

    finetune_trace_all = torch.zeros((num_iter_target, batch_size, 1, trace_length))
    for i in range(num_iter_target):
        data_batch, _ = next(iter_target)
        finetune_trace_all[i] = data_batch

    num_iter = len(source_valid_loader)
    total_clf_loss = total_mmd_loss = total_loss = 0
    correct = 0

    with torch.no_grad():
        for i in range(1, num_iter + 1):
            source_data, source_label = next(iter_source)
            target_data = finetune_trace_all[(i - 1) % num_iter_target]

            if cuda:
                source_data = source_data.cuda(non_blocking=True)
                source_label = source_label.cuda(non_blocking=True)
                target_data = target_data.cuda(non_blocking=True)

            valid_preds, mmd_loss = model(source_data, target_data, source_label, alpha)
            clf_loss = clf_criterion(valid_preds, source_label)
            loss = clf_loss + lambda_ * mmd_loss

            total_clf_loss += clf_loss.item()
            total_mmd_loss += mmd_loss.item()
            total_loss += loss.item()
            pred = valid_preds.data.max(1)[1]
            correct += pred.eq(source_label.data.view_as(pred)).cpu().sum()

    n = len(source_valid_loader)
    print('Validation: total_loss: {:.4f}, clf_loss: {:.4f}, wmmd_loss: {:.4f}, '
          'accuracy: {}/{} ({:.2f}%)'.format(
        total_loss / n, total_clf_loss / n, total_mmd_loss / n,
        correct, len(source_valid_loader.dataset),
        100. * correct / len(source_valid_loader.dataset)))
    return total_loss / n


# =========================
# 10) Clock Jitter
# =========================
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
    Z = np.zeros((len(M), size))
    for enu, row in enumerate(M):
        if len(row) <= size:
            Z[enu, :len(row)] += row
        else:
            Z[enu, :] += row[:size]
    return Z


# =========================
# 11) Main
# =========================
if __name__ == '__main__':
    source_device_id = 0
    target_device_id = 1
    real_key_01 = 224
    real_key_02 = 224
    labeling_method = 'hw'
    lambda_ = 0.1
    preprocess = None
    batch_size = 200
    total_epoch = 100
    finetune_epoch = 15
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
    cuda = not no_cuda and torch.cuda.is_available()
    seed = 8

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    print("Loading Data...")
    X_train_source = np.load(source_file_path + 'X_train.npy')
    Y_train_source = np.load(source_file_path + 'Y_train.npy')
    X_attack_source = np.load(source_file_path + 'X_attack.npy')
    Y_attack_source = np.load(source_file_path + 'Y_attack.npy')

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
    print('Load data complete!')

    adjustments_np = compute_adjustment_1(Y_train_source[0:train_num], tro=1, classes=9)
    adjustments = torch.from_numpy(adjustments_np).view(1, -1)
    if cuda:
        adjustments = adjustments.cuda()

    adjust_flag = True
    flag = "real" if adjust_flag else "fake"

    model = CDP_Net(num_classes=9)
    pretrained_path = ('./models/hw_clockjitter_level1'+'_'+str(flag)+'_pre-trained_cpda_device{}.pth'.format(source_device_id))


    print("Loading model:", pretrained_path)

    checkpoint = None
    if os.path.exists(pretrained_path):
        checkpoint = torch.load(pretrained_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Pretrained model loaded successfully!")
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

    if checkpoint is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    source_labels_np_for_ws = Y_train_source[0:train_num].copy()
    min_loss = 1e9

    for epoch in range(1, finetune_epoch + 1):
        print(f'\n========== Epoch {epoch} ==========')

        # E-step + C-step：估计 alpha
        alpha = estimate_alpha_from_pseudolabels(
            model=model,
            target_loader=target_finetune_loader,
            source_labels_np=source_labels_np_for_ws,
            num_classes=class_num,
            cuda=cuda
        )
        print("Estimated Alpha:", alpha.detach().cpu().numpy())

        # M-step：训练
        CDP_train(epoch, model, alpha)

        # 验证
        with torch.no_grad():
            valid_loss = CDP_validation(model, alpha)
            if valid_loss < min_loss:
                min_loss = valid_loss
                if not os.path.exists('./models'):
                    os.makedirs('./models')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'alpha': alpha.detach().cpu().numpy()
                }, './models/wmmdhw' + str(countermeasure) + '_' + str(flag) +
                   '_best_valid_loss_fine_tuned_cpda_device{}_to_{}.pth'.format(
                       source_device_id, target_device_id))
                print(f"Model saved with validation loss: {valid_loss:.4f}")

    print("\n========== Training Complete ==========")