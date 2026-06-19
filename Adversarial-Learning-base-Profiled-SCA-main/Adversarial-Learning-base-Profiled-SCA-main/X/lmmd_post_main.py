import os
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import gc
import random
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.benchmark = True


# =========================
# 0) Reproducibility
# =========================
def seed_everything(seed: int = 8):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    loader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    return loader


def load_testing(batch_size, kwargs):
    data = TorchDataset(**kwargs)
    loader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    return loader


# =========================
# 2) LMMD Loss
# =========================
class LMMD_loss(nn.Module):
    """
    Local Maximum Mean Discrepancy (LMMD)
    参考论文：Deep Subdomain Adaptation Network (DSAN)
    """

    def __init__(self, class_num=9, kernel_mul=2.0, kernel_num=5, fix_sigma=None,
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
            bandwidth = torch.sum(L2_distance.detach()) / (n_samples ** 2 - n_samples + 1e-12)

        bandwidth = bandwidth / (self.kernel_mul ** (self.kernel_num // 2))
        bandwidth_list = [bandwidth * (self.kernel_mul ** i) for i in range(self.kernel_num)]
        kernel_val = [torch.exp(-L2_distance / (bw + 1e-12)) for bw in bandwidth_list]

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

        if torch.isnan(loss) or torch.isinf(loss):
            return torch.zeros((), device=device)

        return loss


# =========================
# 3) Model (仅 LMMD)
# =========================
class CDP_Net(nn.Module):
    def __init__(self, num_classes=9):
        super(CDP_Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=1),
            nn.SELU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(16, 32, kernel_size=50),
            nn.SELU(),
            nn.AvgPool1d(kernel_size=50, stride=50),
            nn.Conv1d(32, 64, kernel_size=3),
            nn.SELU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        self.classifier_1 = nn.Sequential(
            nn.Linear(64, 20),
            nn.ReLU(inplace=True),
        )
        self.final_classifier = nn.Sequential(
            nn.Linear(20, num_classes)
        )

        # LMMD loss
        self.lmmd_loss = LMMD_loss(
            class_num=num_classes,
            use_confidence_filter=False,
            confidence_threshold=0.3
        )

    def forward(self, source, target, s_label):
        """
        Args:
            source: [bs, 1, L]
            target: [bt, 1, L]
            s_label: [bs] 源域标签

        Returns:
            logits: [bs, C]
            lmmd_loss: scalar
        """
        # ===== 源域 =====
        source_feat = self.features(source)
        source_0 = source_feat.view(source_feat.size(0), -1)  # [bs, 64]
        source_1 = self.classifier_1(source_0)  # [bs, 20]
        logits = self.final_classifier(source_1)  # [bs, C]

        # ===== 目标域 =====
        target_feat = self.features(target)
        target_0 = target_feat.view(target_feat.size(0), -1)  # [bt, 64]
        target_1 = self.classifier_1(target_0)  # [bt, 20]
        target_logits = self.final_classifier(target_1)  # [bt, C]

        # 目标域软标签（用于 LMMD）
        t_prob = torch.softmax(target_logits.detach(), dim=1)

        # ===== LMMD 损失（两层对齐）=====
        lmmd = self.lmmd_loss.get_loss(source_0, target_0, s_label, t_prob)
        lmmd = lmmd + self.lmmd_loss.get_loss(source_1, target_1, s_label, t_prob)

        return logits, lmmd


# =========================
# 4) Logit adjustment
# =========================
def compute_adjustment_1(Y_profiling, tro, classes=9, eps=1e-12):
    counts = np.bincount(Y_profiling, minlength=classes).astype(np.float64)
    pi = counts / np.maximum(counts.sum(), 1.0)
    adj = np.log(np.power(pi, tro) + eps)
    return adj.astype(np.float32)


# =========================
# 5) Train / Validation
# =========================
def CDP_train(epoch, model):
    """训练一个 epoch"""
    model.eval()
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
        loss = clf_loss + alpha * lmmd_loss_val

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
                    alpha
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

            loss = clf_loss + alpha * lmmd_loss_val

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
# 6) Main
# =========================
if __name__ == '__main__':
    seed_everything(8)

    DEVICE_CONFIG = {i: {'key': i, 'folder': f'device{i:02d}'} for i in range(1, 9)}
    source_device_id = 1
    target_device_id = 2

    source_file_path = f"./Data/{DEVICE_CONFIG[source_device_id]['folder']}/"
    target_file_path = f"./Data/{DEVICE_CONFIG[target_device_id]['folder']}/"

    # -----------------
    # hyperparams
    # -----------------
    batch_size = 100
    finetune_epoch = 15
    lr = 0.001
    log_interval = 40

    train_num = 20000
    valid_num = 5000
    source_test_num = 5000
    target_finetune_num = batch_size
    target_test_num = 4500

    trace_offset = 0
    trace_length = 500

    # 损失权重
    alpha = 0.25  # classification vs LMMD (0.5 表示两者权重相等)

    no_cuda = False
    cuda = (not no_cuda) and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    # -----------------
    # load data
    # -----------------
    X_train_source = np.load(source_file_path + 'X_train.npy')
    Y_train_source = np.load(source_file_path + 'Y_train.npy')
    X_attack_source = np.load(source_file_path + 'X_attack.npy')
    Y_attack_source = np.load(source_file_path + 'Y_attack.npy')
    X_attack_target = np.load(target_file_path + 'X_attack.npy')
    Y_attack_target = np.load(target_file_path + 'Y_attack.npy')


    def hstd(X, ref=None):
        if ref is None:
            ref = X
        mn = np.repeat(np.mean(ref, axis=1, keepdims=True), X.shape[1], axis=1)
        std = np.repeat(np.std(ref, axis=1, keepdims=True), X.shape[1], axis=1)
        return (X - mn) / (std + 1e-12)


    X_train_source = hstd(X_train_source, X_train_source)
    X_attack_source = hstd(X_attack_source, X_attack_source)
    X_attack_target = hstd(X_attack_target, X_attack_target)

    # logit adjustment
    adjustments_np = compute_adjustment_1(Y_train_source[0:train_num], tro=1, classes=9)
    adjustments = torch.from_numpy(adjustments_np).view(1, -1).to(device)

    # -----------------
    # dataloaders
    # -----------------
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
    kwargs_source_test = {
        'trs_file': X_attack_source,
        'label_file': Y_attack_source,
        'trace_num': source_test_num,
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
    kwargs_target = {
        'trs_file': X_attack_target[target_finetune_num:target_finetune_num + target_test_num, :],
        'label_file': Y_attack_target[target_finetune_num:target_finetune_num + target_test_num],
        'trace_num': target_test_num,
        'trace_offset': trace_offset,
        'trace_length': trace_length,
    }

    source_train_loader = load_training(batch_size, kwargs_source_train, drop_last=True)
    source_valid_loader = load_training(batch_size, kwargs_source_valid, drop_last=True)
    source_test_loader = load_testing(batch_size, kwargs_source_test)

    # finetune loader 不要 drop_last
    target_finetune_loader = load_training(batch_size, kwargs_target_finetune, drop_last=False)
    target_test_loader = load_testing(batch_size, kwargs_target)

    print('Load data complete!')
    print("len(source_train_loader) =", len(source_train_loader))
    print("len(target_finetune_loader) =", len(target_finetune_loader))

    # -----------------
    # model & optimizer
    # -----------------
    CDP_model = CDP_Net(num_classes=9).to(device)
    print('Construct model complete')

    # 加载预训练模型（使用 strict=False）
    adjust_flag = True

    if adjust_flag:
        flag = "real"
    else:
        flag = "fake"
    pretrained_path = './models/'+str(flag)+'_pre-trained_cpda_device{}.pth'.format(source_device_id)
    print(pretrained_path)
    checkpoint = torch.load(pretrained_path, map_location=device)
    CDP_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print('✅ Loaded pretrained model')

    optimizer = optim.Adam([
        {'params': CDP_model.features.parameters()},
        {'params': CDP_model.classifier_1.parameters()},
        {'params': CDP_model.final_classifier.parameters()}
    ], lr=lr)

    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('✅ Loaded optimizer state')

    # -----------------
    # finetune
    # -----------------
    print("\n" + "=" * 60)
    print("Starting Fine-tuning with LMMD...")
    print(f"alpha={alpha} (classification vs LMMD)")
    print(f"adjust_flag={adjust_flag} (logit adjustment in training)")
    print("=" * 60 + "\n")

    start_time = time.time()

    min_loss = 1e9
    for epoch in range(1, finetune_epoch + 1):
        print(f'\n📌 Train Epoch {epoch}:')
        CDP_train(epoch, CDP_model)

        with torch.no_grad():
            valid_loss = CDP_validation(CDP_model)
            if valid_loss < min_loss:
                min_loss = valid_loss
                if not os.path.exists('./models'):
                    os.makedirs('./models')

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': CDP_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'min_loss': min_loss
                }, './models/lmmd_true'+str(target_finetune_num)+'_best_valid_loss_finetuned_device{}_to_{}.pth'.format(
                    source_device_id, target_device_id
                ))
                print(f'✅ Best model saved at epoch {epoch} with validation loss: {valid_loss:.4f}')

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n总训练时间: {elapsed_time:.2f} 秒")
    # 如果需要更友好的格式，可以转换为分钟和秒
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60
    print(f"总训练时间: {minutes} 分 {seconds:.2f} 秒")



    print("\n" + "=" * 60)
    print("✅ Fine-tuning completed!")
    print(f"Best validation loss: {min_loss:.4f}")
    print("=" * 60)

    # cleanup
    del source_train_loader
    del source_valid_loader
    del target_finetune_loader

    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    gc.collect()
    print("Cleanup complete, exiting...")