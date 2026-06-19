import os
from torch.utils.data import Dataset, DataLoader
import torch
from torch import optim
import numpy as np
from torch import nn
import random
import torch.nn.functional as F

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
# 2) AES tables (kept)
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
           78, 8, 46, 161, 102, 40, 217, 36, 178, 118, 91, 162, 73, 109, 139, 209, 37, 114, 248, 246, 100, 134, 104, 152,
           22, 212, 164, 92, 204, 93, 101, 182, 146, 108, 112, 72, 80, 253, 237, 185, 218, 94, 21, 70, 87, 167, 141, 157,
           132, 144, 216, 171, 0, 140, 188, 211, 10, 247, 228, 88, 5, 184, 179, 69, 6, 208, 44, 30, 143, 202, 63, 15, 2,
           193, 175, 189, 3, 1, 19, 138, 107, 58, 145, 17, 65, 79, 103, 220, 234, 151, 242, 207, 206, 240, 180, 230, 115,
           150, 172, 116, 34, 231, 173, 53, 133, 226, 249, 55, 232, 28, 117, 223, 110, 71, 241, 26, 113, 29, 41, 197, 137,
           111, 183, 98, 14, 170, 24, 190, 27, 252, 86, 62, 75, 198, 210, 121, 32, 154, 219, 192, 254, 120, 205, 90, 244,
           31, 221, 168, 51, 136, 7, 199, 49, 177, 18, 16, 89, 39, 128, 236, 95, 96, 81, 127, 169, 25, 181, 74, 13, 45,
           229, 122, 159, 147, 201, 156, 239, 160, 224, 59, 77, 174, 42, 245, 176, 200, 235, 187, 60, 131, 83, 153, 97,
           23, 43, 4, 126, 186, 119, 214, 38, 225, 105, 20, 99, 85, 33, 12, 125]

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
# 3) MMD / MKWJDAN losses
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
        bandwidth = torch.sum(L2_distance) / (n_samples ** 2 - n_samples + 1e-12)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / (bandwidth_temp + 1e-12)) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)


def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    支持 source/target 样本数不一致的 MMD 估计
    """
    m = int(source.size(0))
    n = int(target.size(0))

    kernels = guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    XX = kernels[:m, :m]        # m x m
    YY = kernels[m:, m:]        # n x n
    XY = kernels[:m, m:]        # m x n
    YX = kernels[m:, :m]        # n x m

    loss = (XX.sum() / (m * m + 1e-12) +
            YY.sum() / (n * n + 1e-12) -
            XY.sum() / (m * n + 1e-12) -
            YX.sum() / (n * m + 1e-12))
    return loss


def conditional_mmd_rbf(source_feat, source_label, target_feat, target_pseudo, num_classes, kernel_mul=2.0, kernel_num=5):
    """
    MKCMMD / CDA: class-wise MMD, average over classes that appear in BOTH domains.
    """
    device_ = source_feat.device
    cda = torch.tensor(0.0, device=device_)
    valid = 0

    for c in range(num_classes):
        s_mask = (source_label == c)
        t_mask = (target_pseudo == c)
        if s_mask.any() and t_mask.any():
            s_c = source_feat[s_mask]
            t_c = target_feat[t_mask]
            cda = cda + mmd_rbf(s_c, t_c, kernel_mul=kernel_mul, kernel_num=kernel_num)
            valid += 1

    if valid == 0:
        return torch.tensor(0.0, device=device_)
    return cda / valid


def dy_pseudo_rectification(pseudo_label, pred_deep, pred_shallow):
    """
    MKWJDAN rectification:
    - CE(pred_deep, pseudo) weighted by exp(-JS(pred_deep, pred_shallow)) + JS
    """
    ce_vec = F.cross_entropy(pred_deep, pseudo_label, reduction="none")  # (B,)
    p = F.softmax(pred_deep, dim=1)
    q = F.softmax(pred_shallow, dim=1)
    m = 0.5 * (p + q)
    js = 0.5 * (F.kl_div(p.log(), m, reduction="none").sum(1) +
                F.kl_div(q.log(), m, reduction="none").sum(1))  # (B,)
    weight = torch.exp(-js)
    return torch.mean(ce_vec * weight + js)


def schedule_l(i, num_iter):
    # 跟您给的 MKWJDAN 代码一致的形式（不追求物理意义，只保持实现一致性）
    # l = 3 / (1 + (num_iter - i) / num_iter * np.exp(i / num_iter)) - 1
    i = float(i)
    num_iter = float(max(num_iter, 1))
    l = 3.0 / (1.0 + (num_iter - i) / num_iter * np.exp(i / num_iter)) - 1.0
    return l


# =========================
# 4) Model: CDP_Net + shallow head
# =========================
class CDP_Net(nn.Module):
    """
    在您原本 CDP_Net 基础上加入：
    - forward_feature: 输出中间层特征（用于MDA/CDA）
    - shallow_head: 用较浅层特征产生一个 logits（用于 JS rectification）
    """
    def __init__(self, num_classes=9):
        super(CDP_Net, self).__init__()
        # encoder
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
        )

        # flatten after features
        self.flatten = nn.Flatten()

        # heads (original)
        self.classifier_1 = nn.Sequential(nn.Linear(256, 20), nn.SELU())
        self.classifier_2 = nn.Sequential(nn.Linear(20, 20), nn.SELU())
        self.classifier_3 = nn.Sequential(nn.Linear(20, 20), nn.SELU())
        self.final_classifier = nn.Sequential(nn.Linear(20, num_classes))

        # shallow head: from feature vector (256) -> logits
        self.shallow_head = nn.Sequential(
            nn.Linear(256, 20),
            nn.SELU(),
            nn.Linear(20, num_classes)
        )

    def forward_feature(self, x):
        """
        返回 256-d feature（用于 MDA/CDA）以及深层 logits
        """
        feat_map = self.features(x)                # (B, C, L)
        feat_vec = self.flatten(feat_map)          # (B, 256)
        h1 = self.classifier_1(feat_vec)
        h2 = self.classifier_2(h1)
        h3 = self.classifier_3(h2)
        logits = self.final_classifier(h3)
        return feat_vec, logits

    def forward_shallow(self, x):
        feat_map = self.features(x)
        feat_vec = self.flatten(feat_map)          # (B, 256)
        logits_shallow = self.shallow_head(feat_vec)
        return logits_shallow


# =========================
# 5) Logit adjustment helper
# =========================
def compute_adjustment_1(Y_profiling, tro, classes=9, eps=1e-12):
    counts = np.bincount(Y_profiling, minlength=classes).astype(np.float64)
    pi = counts / np.maximum(counts.sum(), 1.0)
    adjustments = np.log(np.power(pi, tro) + eps)  # log(pi^tau)
    return pi


def calculate_HW(data):
    return HW_byte_np[data.astype(int)]


# =========================
# 6) Train / Validation (MKWJDAN + LA only on source CE)
# =========================
def CDP_train(epoch, model):
    model.train()  # 【改】训练应该用 train()

    iter_source = iter(source_train_loader)
    iter_target = iter(target_finetune_loader)
    num_iter_target = len(target_finetune_loader)

    # 预取 target batches
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

        # ===== Forward =====
        s_feat, s_logits = model.forward_feature(source_data)
        t_feat, t_logits = model.forward_feature(target_data)

        # shallow logits for target (rectification)
        t_logits_shallow = model.forward_shallow(target_data)

        # ===== (1) Source CE with logit adjustment (TRAIN ONLY) =====
        s_logits_ce = s_logits
        if adjust_flag:
            s_logits_ce = s_logits_ce + adjustments  # broadcast (1,C) + (B,C)

        clf_loss = clf_criterion(s_logits_ce, source_label)

        # ===== (2) Pseudo labels from deep target logits =====
        pseudo = torch.argmax(t_logits.detach(), dim=1)

        # ===== (3) Rectification loss (JS-based) =====
        rect_loss = dy_pseudo_rectification(pseudo, t_logits, t_logits_shallow)

        # ===== (4) MDA & CDA on logits (MKWJDAN style) =====
        mda_loss = mmd_rbf(s_logits, t_logits)  # marginal alignment
        cda_loss = conditional_mmd_rbf(
            s_logits, source_label,
            t_logits, pseudo,
            num_classes=class_num
        )

        # adaptive weight
        denom = (cda_loss + mda_loss + 1e-12)
        u = cda_loss / denom

        # dynamic coefficient l
        l = schedule_l(i, num_iter)

        # ===== Total loss =====
        # 与 MKWJDAN 一致：clf + l*((1-u)*MDA + u*CDA) + beta*rect
        loss = clf_loss + lambda_ * l * ((1.0 - u) * mda_loss + u * cda_loss) + rect_beta * rect_loss

        loss.backward()
        optimizer.step()

        if i % log_interval == 0:
            print('Train Epoch {}: [{}/{} ({:.0f}%)]\ttotal_loss: {:.6f}\tclf_loss: {:.6f}\t'
                  'mda_loss: {:.6f}\tcda_loss: {:.6f}\trect_loss: {:.6f}\tu: {:.4f}\tl: {:.4f}'.format(
                epoch, i * len(source_data), len(source_train_loader) * batch_size,
                100. * i / len(source_train_loader),
                loss.item(), clf_loss.item(), mda_loss.item(), cda_loss.item(), rect_loss.item(),
                float(u.detach().cpu()), float(l)
            ))


def CDP_validation(model):
    """
    验证阶段：严格不使用 logit adjustment（您要求）
    """
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
    total_clf_loss, total_mda, total_cda, total_rect, total_loss = 0, 0, 0, 0, 0
    correct = 0

    with torch.no_grad():
        for i in range(1, num_iter + 1):
            source_data, source_label = next(iter_source)
            target_data = finetune_trace_all[(i - 1) % num_iter_target, :, :, :]

            if cuda:
                source_data = source_data.cuda(non_blocking=True)
                source_label = source_label.cuda(non_blocking=True)
                target_data = target_data.cuda(non_blocking=True)

            s_feat, s_logits = model.forward_feature(source_data)
            t_feat, t_logits = model.forward_feature(target_data)
            t_logits_shallow = model.forward_shallow(target_data)

            # 【关键】Validation 不做 logit adjustment
            clf_loss = clf_criterion(s_logits, source_label)

            pseudo = torch.argmax(t_logits, dim=1)
            rect_loss = dy_pseudo_rectification(pseudo, t_logits, t_logits_shallow)

            mda_loss = mmd_rbf(s_logits, t_logits)
            cda_loss = conditional_mmd_rbf(s_logits, source_label, t_logits, pseudo, num_classes=class_num)

            denom = (cda_loss + mda_loss + 1e-12)
            u = cda_loss / denom
            l = schedule_l(i, num_iter)

            loss = clf_loss + lambda_ * l * ((1.0 - u) * mda_loss + u * cda_loss) + rect_beta * rect_loss

            total_clf_loss += clf_loss.item()
            total_mda += mda_loss.item()
            total_cda += cda_loss.item()
            total_rect += rect_loss.item()
            total_loss += loss.item()

            pred = s_logits.data.max(1)[1]
            correct += pred.eq(source_label.data.view_as(pred)).cpu().sum()

    n = len(source_valid_loader)
    total_loss /= n
    total_clf_loss /= n
    total_mda /= n
    total_cda /= n
    total_rect /= n

    print('Validation: total_loss: {:.4f}, clf_loss: {:.4f}, mda_loss: {:.4f}, cda_loss: {:.4f}, rect_loss: {:.4f}, '
          'accuracy: {}/{} ({:.2f}%)'.format(
        total_loss, total_clf_loss, total_mda, total_cda, total_rect,
        correct, len(source_valid_loader.dataset),
        100. * correct / len(source_valid_loader.dataset)
    ))
    return total_loss


# =========================
# 7) Clock jitter (kept)
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
    labeling_method = 'hw'

    # ===== MKWJDAN params =====
    lambda_ = 0.1          # 对齐损失系数（您原本就有）
    rect_beta = 0.1        # rectification 权重（对应您MKWJDAN代码里的 0.1*loss）
    # =========================

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
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    X_train_source = np.load(source_file_path + 'X_train.npy')
    Y_train_source = np.load(source_file_path + 'Y_train.npy')
    X_attack_source = np.load(source_file_path + 'X_attack.npy')
    Y_attack_source = np.load(source_file_path + 'Y_attack.npy')

    # add clock jitter to the target domain
    X_attack_target = addClockJitter(X_attack_source, clock_range, trace_length)
    Y_attack_target = Y_attack_source

    if labeling_method == 'identity':
        class_num = 256
    elif labeling_method == 'hw':
        class_num = 9
        Y_train_source = calculate_HW(Y_train_source)
        Y_attack_source = calculate_HW(Y_attack_source)
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

    # ===== Logit adjustment (TRAIN ONLY usage) =====
    adjustments_np = compute_adjustment_1(Y_attack_target[0:target_finetune_num], tro=1, classes=class_num)
    adjustments = torch.from_numpy(adjustments_np).view(1, -1)
    if cuda:
        adjustments = adjustments.cuda()
    adjust_flag = True  # 您说只用于源域训练CE
    flag = "real" if adjust_flag else "fake"

    # ===== Model =====
    model = CDP_Net(num_classes=class_num)
    pretrained_path = ('./models/' + str(countermeasure) + '_' + str(flag) +
                       '_pre-trained_cpda_device{}.pth'.format(source_device_id))
    print("Loading model:", pretrained_path)

    checkpoint = None
    if os.path.exists(pretrained_path):
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        model_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(model_dict, strict=False)
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

    # load optimizer state if available
    if checkpoint is not None and isinstance(checkpoint, dict) and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception as e:
            print("Warning: optimizer state not loaded:", e)

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
                }, './models/new' + str(countermeasure) + '_' + str(flag) +
                   '_best_valid_loss_fine_tuned_mkwjdan_device{}_to_{}.pth'.format(source_device_id, target_device_id))
