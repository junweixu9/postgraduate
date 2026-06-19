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
        data, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    return train_loader


def load_testing(batch_size, kwargs):
    data = TorchDataset(**kwargs)
    test_loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=False, drop_last=True,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    return test_loader


# =========================
# 2) AES helper (HW labels)
# =========================
HW_byte = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
           1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
           1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
           2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
           1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
           2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
           2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
           3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8]
HW_byte_np = np.array(HW_byte, dtype=np.int32)

def calculate_HW(data):
    return HW_byte_np[data.astype(int)]


def compute_adjustment_1(Y_profiling, tro, classes=9, eps=1e-12):
    counts = np.bincount(Y_profiling, minlength=classes).astype(np.float64)
    pi = counts / np.maximum(counts.sum(), 1.0)
    adjustments = np.log(np.power(pi, tro) + eps)  # log(pi^tau)
    return adjustments.astype(np.float32)


# =========================
# 3) JMMD utilities
# =========================
def gaussian_kernel_matrix(x, y, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    x: (bs, d), y: (bt, d)
    return: (bs+bt, bs+bt) kernel matrix on concatenated samples
    """
    total = torch.cat([x, y], dim=0)  # (n, d)
    n_samples = total.size(0)

    total0 = total.unsqueeze(0).expand(n_samples, n_samples, total.size(1))
    total1 = total.unsqueeze(1).expand(n_samples, n_samples, total.size(1))
    L2_distance = ((total0 - total1) ** 2).sum(2)

    if fix_sigma is not None:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance) / (n_samples ** 2 - n_samples + 1e-12)

    bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / (bw + 1e-12)) for bw in bandwidth_list]
    return sum(kernel_val)


def mmd_from_kernel(K, bs):
    """
    K: (n,n) kernel matrix where first bs are source, last are target
    """
    XX = K[:bs, :bs]
    YY = K[bs:, bs:]
    XY = K[:bs, bs:]
    YX = K[bs:, :bs]
    return torch.mean(XX + YY - XY - YX)


def jmmd_loss(source_list, target_list, kernel_mul=2.0, kernel_num=5, fix_sigma_list=None):
    """
    source_list: list of tensors [f1_s, f2_s, ..., p_s] each (bs, d_k)
    target_list: list of tensors [f1_t, f2_t, ..., p_t] each (bs, d_k)
    JMMD via product kernel: K_joint = Π_k K_k, then MMD(K_joint).
    (Common practice from JAN-style JMMD) :contentReference[oaicite:4]{index=4}
    """
    assert len(source_list) == len(target_list)
    L = len(source_list)
    bs = source_list[0].size(0)

    if fix_sigma_list is None:
        fix_sigma_list = [None] * L
    assert len(fix_sigma_list) == L

    K_joint = None
    for k in range(L):
        xs = source_list[k]
        xt = target_list[k]
        # ensure 2D
        xs = xs.view(xs.size(0), -1)
        xt = xt.view(xt.size(0), -1)
        Kk = gaussian_kernel_matrix(xs, xt, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma_list[k])
        K_joint = Kk if K_joint is None else (K_joint * Kk)  # Hadamard product

    return mmd_from_kernel(K_joint, bs)


# =========================
# 4) Model (return multi-layer reps for JMMD)
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
        self.classifier_1 = nn.Sequential(
            nn.Linear(256, 20),
            nn.SELU(),
        )
        self.classifier_2 = nn.Sequential(
            nn.Linear(20, 20),
            nn.SELU(),
        )
        self.classifier_3 = nn.Sequential(
            nn.Linear(20, 20),
            nn.SELU(),
        )
        # the output layer
        self.final_classifier = nn.Sequential(
            nn.Linear(20, num_classes)
        )

    def encode_layers(self, x):
        z0 = self.features(x)                  # (bs, 256)
        z1 = self.classifier_1(z0)             # (bs, 20)
        z2 = self.classifier_2(z1)             # (bs, 20)
        z3 = self.classifier_3(z2)             # (bs, 20)
        logits = self.final_classifier(z3)     # (bs, C)
        probs = torch.softmax(logits, dim=1)   # soft labels for JMMD
        return z0, z1, z2, z3, logits, probs

    def forward(self, source, target, jmmd_enable=True, jmmd_layers=("z0", "z1", "z2", "prob"),
                kernel_mul=2.0, kernel_num=5):
        # source forward
        s_z0, s_z1, s_z2, s_z3, s_logits, s_prob = self.encode_layers(source)
        # target forward
        t_z0, t_z1, t_z2, t_z3, t_logits, t_prob = self.encode_layers(target)

        if not jmmd_enable:
            jmmd = torch.tensor(0.0, device=source.device)
        else:
            # choose which reps join in JMMD
            def pick(z0, z1, z2, z3, prob):
                out = []
                for name in jmmd_layers:
                    if name == "z0":
                        out.append(z0)
                    elif name == "z1":
                        out.append(z1)
                    elif name == "z2":
                        out.append(z2)
                    elif name == "z3":
                        out.append(z3)
                    elif name in ("p", "prob", "y"):
                        out.append(prob)
                    else:
                        raise ValueError(f"Unknown jmmd layer key: {name}")
                return out

            s_list = pick(s_z0, s_z1, s_z2, s_z3, s_prob)
            t_list = pick(t_z0, t_z1, t_z2, t_z3, t_prob)
            jmmd = jmmd_loss(s_list, t_list, kernel_mul=kernel_mul, kernel_num=kernel_num)

        return s_logits, jmmd


# =========================
# 5) Train / Validation
# =========================
def CDP_train(epoch, model):
    model.train()  # ★训练要用 train()

    iter_source = iter(source_train_loader)
    iter_target = iter(target_finetune_loader)
    num_iter_target = len(target_finetune_loader)

    # prefetch target
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

        # ★ 用 JMMD（联合多层 + soft label）
        source_logits, jmmd = model(
            source_data, target_data,
            jmmd_enable=True,
            jmmd_layers=("z0", "z1", "z2", "prob"),  # 你也可改为 ("z1","z2","z3","prob") 等
            kernel_mul=2.0, kernel_num=5
        )

        # ★ 训练期间：允许 logit adjustment（按您原逻辑）
        if adjust_flag:
            source_logits = source_logits + 1.0 * adjustments

        clf_loss = clf_criterion(source_logits, source_label)
        loss = clf_loss + lambda_ * jmmd

        loss.backward()
        optimizer.step()

        if i % log_interval == 0:
            print('Train Epoch {}: [{}/{} ({:.0f}%)]\ttotal_loss: {:.6f}\tclf_loss: {:.6f}\tjmmd_loss: {:.6f}'.format(
                epoch, i * len(source_data), len(source_train_loader) * batch_size,
                100. * i / len(source_train_loader), loss.item(), clf_loss.item(), jmmd.item()
            ))


def CDP_validation(model):
    # ★ 验证期间：明确不使用 logit adjustment（按您的要求）
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
    total_jmmd_loss = 0.0
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

            valid_logits, jmmd = model(
                source_data, target_data,
                jmmd_enable=True,
                jmmd_layers=("z0", "z1", "z2", "prob"),
                kernel_mul=2.0, kernel_num=5
            )

            # ★ 不加 adjustments
            clf_loss = clf_criterion(valid_logits, source_label)
            loss = clf_loss + lambda_ * jmmd

            total_clf_loss += clf_loss.item()
            total_jmmd_loss += jmmd.item()
            total_loss += loss.item()

            pred = valid_logits.data.max(1)[1]
            correct += pred.eq(source_label.data.view_as(pred)).cpu().sum()

    total_loss /= len(source_valid_loader)
    total_clf_loss /= len(source_valid_loader)
    total_jmmd_loss /= len(source_valid_loader)

    print('Validation: total_loss: {:.4f}, clf_loss: {:.4f}, jmmd_loss: {:.4f}, accuracy: {}/{} ({:.2f}%)'.format(
        total_loss, total_clf_loss, total_jmmd_loss, correct, len(source_valid_loader.dataset),
        100. * correct / len(source_valid_loader.dataset)
    ))
    return total_loss


# =========================
# 6) Clock jitter (unchanged)
# =========================
def regulateMatrix(M, size):
    maxlen = size
    Z = np.zeros((len(M), maxlen))
    for enu, row in enumerate(M):
        if len(row) <= maxlen:
            Z[enu, :len(row)] += row
        else:
            Z[enu, :] += row[:maxlen]
    return Z

def addClockJitter(traces, clock_range, trace_length):
    print('Add clock jitters...')
    output_traces = []
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


# =========================
# 7) Main
# =========================
if __name__ == '__main__':
    source_device_id = 0
    target_device_id = 1
    real_key_01 = 224
    real_key_02 = 224
    labeling_method = 'hw'

    lambda_ = 0.1  # penalty coefficient for JMMD
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

    # ===== logit adjustment (train only) =====
    adjustments_np = compute_adjustment_1(Y_train_source[0:train_num], tro=1, classes=9)
    adjustments = torch.from_numpy(adjustments_np).view(1, -1)
    if cuda:
        adjustments = adjustments.cuda()
    adjust_flag = True

    flag = "real" if adjust_flag else "fake"

    model = CDP_Net(num_classes=9)
    pretrained_path = ('./models/' + str(countermeasure) + '_' + str(flag) + '_pre-trained_cpda_device{}.pth'.format(
        source_device_id))
    print("Loading model:", pretrained_path)
    # 【优化】增加文件存在性检查，避免 Crash
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

    # 【修正】仅当 checkpoint 加载成功且包含 optimizer 时才加载状态
    if os.path.exists(pretrained_path) and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    for epoch in range(1, finetune_epoch + 1):
        print(f'Train Epoch {epoch}:')
        CDP_train(epoch, model)

        with torch.no_grad():
            valid_loss = CDP_validation(model)
            if valid_loss < min_loss:
                min_loss = valid_loss
                if not os.path.exists('../SAKURA/models'):
                    os.makedirs('../SAKURA/models')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, './models/' + str(countermeasure) + '_' + str(flag) +
                   '_best_valid_loss_fine_tuned_cpda_device{}_to_{}.pth'.format(source_device_id, target_device_id))
