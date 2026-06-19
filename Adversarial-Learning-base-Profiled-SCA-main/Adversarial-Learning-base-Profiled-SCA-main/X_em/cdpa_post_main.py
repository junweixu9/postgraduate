import os
import time

import requests

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

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    - source : source data
    - target : target data
    - kernel_mul : multiplicative step of bandwidth (sigma)
    - kernel_num : the number of guassian kernels
    - fix_sigma : use a fix value of bandwidth
    """
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    # |x-y|
    L2_distance = ((total0 - total1) ** 2).sum(2)

    # bandwidth
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    # take the current bandwidth as the median value, and get a list of bandwidths (for example, when bandwidth is 1, we get [0.25,0.5,1,2,4]).
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

    # exp(-|x-y|/bandwidth)
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for \
                  bandwidth_temp in bandwidth_list]

    # return the final kernel matrix
    return sum(kernel_val)


### MMD loss function based on guassian kernels
def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    - source : source data
    - target : target data
    - kernel_mul : multiplicative step of bandwidth (sigma)
    - kernel_num : the number of guassian kernels
    - fix_sigma : use a fix value of bandwidth
    """
    loss = 0.0
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]  # Source<->Source
    YY = kernels[batch_size:, batch_size:]  # Target<->Target
    XY = kernels[:batch_size, batch_size:]  # Source<->Target
    YX = kernels[batch_size:, :batch_size]  # Target<->Source
    loss = torch.mean(XX + YY - XY - YX)
    return loss


class CDP_Net(nn.Module):
    def __init__(self, num_classes=9):
        super(CDP_Net, self).__init__()
        # the encoder part
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
        # the fully-connected layer 1
        self.classifier_1 = nn.Sequential(
            nn.Linear(384, 20),
            nn.ReLU(inplace=True),
        )
        # the output layer
        self.final_classifier = nn.Sequential(
            nn.Linear(20, num_classes)
        )

    # how the network runs
    def forward(self, source, target):
        mmd_loss = 0
        # source data flow
        source = self.features(source)
        source_0 = source.view(source.size(0), -1)
        source_1 = self.classifier_1(source_0)

        # target data flow
        target = self.features(target)
        target = target.view(target.size(0), -1)
        mmd_loss += mmd_rbf(source_0, target)
        target = self.classifier_1(target)
        mmd_loss += mmd_rbf(source_1, target)

        result = self.final_classifier(source_1)
        return result, mmd_loss


# =========================
# 6) Logit adjustment
# =========================
def compute_adjustment_1(Y_profiling, tro, classes=9, eps=1e-12):
    counts = np.bincount(Y_profiling, minlength=classes).astype(np.float64)
    pi = counts / np.maximum(counts.sum(), 1.0)
    adj = np.log(np.power(pi, tro) + eps)
    return adj.astype(np.float32)

def CDP_train(epoch, model):
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
        target_data = finetune_trace_all[(i - 1) % num_iter_target]

        source_data = source_data.to(device, non_blocking=True)
        source_label = source_label.to(device, non_blocking=True)
        target_data = target_data.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        source_preds, mmd_loss = model(source_data, target_data)

        # 训练阶段：可选 logit adjustment
        if adjust_flag:
            source_preds = source_preds + adjustments

        clf_loss = clf_criterion(source_preds, source_label)

        loss = clf_loss + lambda_ * mmd_loss

        loss.backward()
        optimizer.step()

        if i % log_interval == 0:
            print('Train Epoch {}: [{}/{} ({:.0f}%)]\ttotal_loss: {:.6f}\tclf_loss: {:.6f}\tmmd_loss: {:.6f}\tlambda: {:.4f}'.format(
                epoch, i * len(source_data), len(source_train_loader.dataset),
                100. * i / len(source_train_loader),
                loss.item(), clf_loss.item(), mmd_loss.item(), 0.1
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
    total_clf_loss = 0
    total_mmd_loss = 0
    total_loss = 0
    correct = 0

    # 【优化】使用 no_grad 上下文
    with torch.no_grad():
        for i in range(1, num_iter + 1):
            source_data, source_label = next(iter_source)
            target_data = finetune_trace_all[(i - 1) % num_iter_target, :, :, :]

            if cuda:
                source_data = source_data.cuda(non_blocking=True)
                source_label = source_label.cuda(non_blocking=True)
                target_data = target_data.cuda(non_blocking=True)

            # 【优化】移除 Variable

            valid_preds, mmd_loss = model(source_data, target_data)
            clf_loss = clf_criterion(valid_preds, source_label)

            loss = clf_loss + lambda_ * mmd_loss

            total_clf_loss += clf_loss.item()
            total_mmd_loss += mmd_loss.item()
            total_loss += loss.item()

            pred = valid_preds.data.max(1)[1]
            correct += pred.eq(source_label.data.view_as(pred)).cpu().sum()

    total_loss /= len(source_valid_loader)
    total_clf_loss /= len(source_valid_loader)
    total_mmd_loss /= len(source_valid_loader)

    print('Validation: total_loss: {:.4f}, clf_loss: {:.4f}, mmd_loss: {:.4f}, accuracy: {}/{} ({:.2f}%)'.format(
        total_loss, total_clf_loss, total_mmd_loss, correct, len(source_valid_loader.dataset),
        100. * correct / len(source_valid_loader.dataset)))

    # 【修正】恢复原始的单个返回值
    return total_loss


# =========================
# 9) Main
# =========================
if __name__ == '__main__':
    seed_everything(8)

    DEVICE_CONFIG = {i: {'key': i, 'folder': f'device{i:02d}'} for i in range(1, 9)}
    source_device_id = 1
    target_device_id = 4

    source_file_path = f"./Data/{DEVICE_CONFIG[source_device_id]['folder']}/"
    target_file_path = f"./Data/{DEVICE_CONFIG[target_device_id]['folder']}/"

    # -----------------
    # hyperparams
    # -----------------
    lambda_ = 0.1  # Penalty coefficient
    labeling_method = 'hw'  # labeling of trace
    preprocess = 'horizontal_standardization'  # preprocess method
    batch_size = 250
    total_epoch = 100
    finetune_epoch = 15  # epoch number for fine-tuning
    lr = 0.001  # learning rate
    log_interval = 20  # epoch interval to log training information
    train_num = 20000
    valid_num = 5000
    source_test_num = 10000
    target_finetune_num = 500
    target_test_num = 9500
    trace_offset = 0
    trace_length = 1500
    no_cuda = False
    cuda = not no_cuda and torch.cuda.is_available()
    seed = 8
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    if labeling_method == 'identity':
        class_num = 256
    elif labeling_method == 'hw':
        class_num = 9
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

    adjust_flag = False

    if adjust_flag:
        flag = "real"
    else:
        flag = "fake"
    path = './models/'+str(flag)+'_pre-trained_cpda_device{}.pth'.format(source_device_id)

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

    # ✅ 关键：finetune loader 不要 drop_last（更稳）
    target_finetune_loader = load_training(batch_size, kwargs_target_finetune, drop_last=False)
    target_test_loader = load_testing(batch_size, kwargs_target)

    print('Load data complete!')
    print("len(source_train_loader) =", len(source_train_loader))
    print("len(target_finetune_loader) =", len(target_finetune_loader))

    # -----------------
    # model & optimizer
    # -----------------
    CDP_model = CDP_Net(num_classes=class_num).to(device)
    print('Construct model complete')

    checkpoint = torch.load(path, map_location=device)
    CDP_model.load_state_dict(checkpoint['model_state_dict'])

    optimizer = optim.Adam([
        {'params': CDP_model.features.parameters()},
        {'params': CDP_model.classifier_1.parameters()},
        {'params': CDP_model.final_classifier.parameters()}
    ], lr=lr)

    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # -----------------
    # finetune
    # -----------------
    start_time = time.time()

    min_loss = 1e9
    for epoch in range(1, finetune_epoch + 1):
        print(f'Train Epoch {epoch}:')
        CDP_train(epoch, CDP_model)

        with torch.no_grad():
            valid_loss = CDP_validation(CDP_model)
            if valid_loss < min_loss:
                min_loss = valid_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': CDP_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, './models/hw_best_valid_loss_finetuned_device{}_to_{}.pth'.format(source_device_id, target_device_id))
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n总训练时间: {elapsed_time:.2f} 秒")
    # 如果需要更友好的格式，可以转换为分钟和秒
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60
    print(f"总训练时间: {minutes} 分 {seconds:.2f} 秒")


    # cleanup
    del source_train_loader
    del source_valid_loader
    del target_finetune_loader

    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    gc.collect()
    print("Cleanup complete, exiting...")
    ret = requests.get('https://api.day.app/Cqdu3932Dcbe3dduH2EJPM/训练/完成')