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


# ============================================================
# 1) Dataset / Dataloader
# ============================================================
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


# ============================================================
# 2) AES tables + HW
# ============================================================
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


# ============================================================
# 3) Logit Adjustment
# ============================================================
def compute_adjustment_1(Y_profiling, tro, classes=9, eps=1e-12):
    counts = np.bincount(Y_profiling, minlength=classes).astype(np.float64)
    pi = counts / np.maximum(counts.sum(), 1.0)
    adjustments = np.log(np.power(pi, tro) + eps)  # log(pi^tau)
    return adjustments.astype(np.float32)


# ============================================================
# 4) MMD kernels (RBF or Polynomial)
#    - paper emphasizes higher-order moments via polynomial kernel MMD
# ============================================================
def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size(0)) + int(target.size(0))
    total = torch.cat([source, target], dim=0)

    total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
    total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
    L2_distance = ((total0 - total1) ** 2).sum(2)

    if fix_sigma is not None:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance) / (n_samples ** 2 - n_samples + 1e-12)

    bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / (bw + 1e-12)) for bw in bandwidth_list]
    return sum(kernel_val)


def polynomial_kernel_matrix(source, target, degree=2, c=1.0):
    """
    K(x,y) = (x·y + c)^degree
    """
    total = torch.cat([source, target], dim=0)  # [n, d]
    G = total @ total.t()
    K = (G + c) ** degree
    return K


def mmd(source, target, kernel_type="rbf",
        kernel_mul=2.0, kernel_num=5, fix_sigma=None,
        poly_degree=2, poly_c=1.0):
    """
    unbiased-ish version using block means (same as your code style)
    """
    batch_size = int(source.size(0))
    if kernel_type == "rbf":
        kernels = gaussian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    elif kernel_type == "poly":
        kernels = polynomial_kernel_matrix(source, target, degree=poly_degree, c=poly_c)
    else:
        raise ValueError(f"Unknown kernel_type={kernel_type}")

    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss


# ============================================================
# 5) Clock jitter (keep yours)
# ============================================================
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


# ============================================================
# 6) AAAI19-style Unilateral Residual Transform
#    F(x) = (I + A) x  -> implemented as x + Linear(x) (Linear init to zero)
#    and add ||A||^2 regularization
# ============================================================
class UnilateralResidualTransform(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.A = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.A.weight)  # start from identity: F(x)=x

    def forward(self, x):
        return x + self.A(x)

    def reg_loss(self):
        # Frobenius norm squared of A
        return torch.sum(self.A.weight ** 2)


# ============================================================
# 7) Network (insert unilateral+residual before MMD)
# ============================================================
class CDP_Net(nn.Module):
    def __init__(self, num_classes=9,
                 kernel_type="rbf",           # "rbf" or "poly"
                 poly_degree=2, poly_c=1.0,
                 kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super(CDP_Net, self).__init__()

        self.kernel_type = kernel_type
        self.poly_degree = poly_degree
        self.poly_c = poly_c
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma

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
            nn.SELU()
        )
        self.classifier_3 = nn.Sequential(
            nn.Linear(20, 20),
            nn.SELU()
        )
        self.final_classifier = nn.Sequential(
            nn.Linear(20, num_classes)
        )

        # --- Unilateral residual transforms for each aligned layer ---
        # Align source->target at multiple levels: source_0(256), source_1(20), source_2(20)
        self.ut0 = UnilateralResidualTransform(dim=256)
        self.ut1 = UnilateralResidualTransform(dim=20)
        self.ut2 = UnilateralResidualTransform(dim=20)

    def forward(self, source, target):
        # ----- source flow -----
        source_feat = self.features(source)
        source_0 = source_feat.view(source_feat.size(0), -1)     # [B,256]
        source_1 = self.classifier_1(source_0)                   # [B,20]
        source_2 = self.classifier_2(source_1)                   # [B,20]
        source_3 = self.classifier_3(source_2)                   # [B,20]
        logits = self.final_classifier(source_3)

        # ----- target flow (only used for MMD) -----
        target_feat = self.features(target)
        target_0 = target_feat.view(target_feat.size(0), -1)     # [B,256]
        target_1 = self.classifier_1(target_0)                   # [B,20]
        target_2 = self.classifier_2(target_1)                   # [B,20]

        # ----- AAAI19 unilateral + residual: transform ONLY source representations -----
        # MMD( F(source), target )
        mmd_loss = 0.0
        mmd_loss = mmd_loss + mmd(
            self.ut0(source_0), target_0,
            kernel_type=self.kernel_type,
            kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma,
            poly_degree=self.poly_degree, poly_c=self.poly_c
        )
        mmd_loss = mmd_loss + mmd(
            self.ut1(source_1), target_1,
            kernel_type=self.kernel_type,
            kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma,
            poly_degree=self.poly_degree, poly_c=self.poly_c
        )
        mmd_loss = mmd_loss + mmd(
            self.ut2(source_2), target_2,
            kernel_type=self.kernel_type,
            kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma,
            poly_degree=self.poly_degree, poly_c=self.poly_c
        )

        # regularization term for A matrices (||A||^2)
        reg = self.ut0.reg_loss() + self.ut1.reg_loss() + self.ut2.reg_loss()

        return logits, mmd_loss, reg


# ============================================================
# 8) Train / Validation
# ============================================================
def CDP_train(epoch, model):
    model.train()  # 【修正】训练要 train()

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

        source_preds, mmd_loss, reg_loss = model(source_data, target_data)

        if adjust_flag:
            source_preds = source_preds + adjustments  # broadcast [1,C] to [B,C]

        clf_loss = clf_criterion(source_preds, source_label)

        # total loss = clf + lambda*mmd + lambda_reg*||A||^2
        loss = clf_loss + lambda_ * mmd_loss + lambda_reg * reg_loss

        loss.backward()
        optimizer.step()

        if i % log_interval == 0:
            print('Train Epoch {}: [{}/{} ({:.0f}%)]\t'
                  'total_loss: {:.6f}\tclf_loss: {:.6f}\tmmd_loss: {:.6f}\treg: {:.6f}'.format(
                      epoch, i * len(source_data), len(source_train_loader) * batch_size,
                      100. * i / len(source_train_loader),
                      loss.item(), clf_loss.item(), mmd_loss.item(), reg_loss.item()
                  ))


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
    total_reg = 0.0
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

            valid_preds, mmd_loss, reg_loss = model(source_data, target_data)

            clf_loss = clf_criterion(valid_preds, source_label)
            loss = clf_loss + lambda_ * mmd_loss + lambda_reg * reg_loss

            total_clf_loss += clf_loss.item()
            total_mmd_loss += mmd_loss.item()
            total_reg += reg_loss.item()
            total_loss += loss.item()

            pred = valid_preds.data.max(1)[1]
            correct += pred.eq(source_label.data.view_as(pred)).cpu().sum()

    total_loss /= len(source_valid_loader)
    total_clf_loss /= len(source_valid_loader)
    total_mmd_loss /= len(source_valid_loader)
    total_reg /= len(source_valid_loader)

    print('Validation: total_loss: {:.4f}, clf_loss: {:.4f}, mmd_loss: {:.4f}, reg: {:.4f}, '
          'accuracy: {}/{} ({:.2f}%)'.format(
              total_loss, total_clf_loss, total_mmd_loss, total_reg,
              correct, len(source_valid_loader.dataset),
              100. * correct / len(source_valid_loader.dataset)
          ))

    return total_loss


# ============================================================
# 9) Main
# ============================================================
if __name__ == '__main__':
    source_device_id = 0
    target_device_id = 1
    real_key_01 = 224
    real_key_02 = 224
    labeling_method = 'hw'

    lambda_ = 0.1        # MMD penalty
    lambda_reg = 1e-4    # 【新增】AAAI19 residual transform regularization ||A||^2 (可调)

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
    adjustments_np = compute_adjustment_1(Y_train_source[0:train_num], tro=1, classes=9)
    adjustments = torch.from_numpy(adjustments_np).view(1, -1)
    if cuda:
        adjustments = adjustments.cuda()

    adjust_flag = True
    flag = "real" if adjust_flag else "fake"

    # 【关键】你可以切换核：kernel_type="poly" 更贴近AAAI19“高阶矩匹配”
    # 推荐先试 poly_degree=2 或 3
    model = CDP_Net(
        num_classes=9,
        kernel_type="poly",   # "rbf" / "poly"
        poly_degree=2,
        poly_c=1.0
    )

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
        {'params': model.final_classifier.parameters()},
        {'params': model.ut0.parameters()},
        {'params': model.ut1.parameters()},
        {'params': model.ut2.parameters()},
    ], lr=lr)

    if cuda:
        model.cuda()

    min_loss = 1e9
    if checkpoint is not None and isinstance(checkpoint, dict) and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Loaded optimizer state.")
        except ValueError as e:
            print(f"Skip loading optimizer state due to mismatch: {e}")

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
                }, './models/' + str(countermeasure) + '_' + str(flag) +
                   '_best_valid_loss_fine_tuned_cpda_device{}_to_{}.pth'.format(
                       source_device_id, target_device_id
                   ))
