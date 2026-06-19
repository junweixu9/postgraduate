import os
import time

from torch.utils.data import Dataset, DataLoader
import torch
from torch import optim
import numpy as np
from torch import nn
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
        # 【优化】直接转换 Tensor，移除 transforms 和不必要的 reshape
        trace_tensor = torch.from_numpy(trace).float().unsqueeze(0)
        label_tensor = torch.tensor(self.label_file[index], dtype=torch.long)
        return trace_tensor, label_tensor

    def __len__(self):
        return self.trace_num


### data loader for training
def load_training(batch_size, kwargs):
    data = TorchDataset(**kwargs)
    # 【优化】开启多进程预取和持久化进程
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True,
                                               drop_last=True, num_workers=4,
                                               pin_memory=True, persistent_workers=True)
    return train_loader


### data loader for testing
def load_testing(batch_size, kwargs):
    data = TorchDataset(**kwargs)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False,
                                              drop_last=True, num_workers=4,
                                              pin_memory=True, persistent_workers=True)
    return test_loader

HW_byte = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2,
           3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2, 3,
           3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3,
           4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4,
           3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5,
           6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4,
           4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5,
           6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8]

# 【优化】将 HW_byte 转换为 Numpy 数组，支持向量化索引
HW_byte_np = np.array(HW_byte, dtype=np.int32)


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
        # data.sum() instead of sum(data)
        bandwidth = torch.sum(L2_distance) / (n_samples ** 2 - n_samples)

    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

    # exp(-|x-y|/bandwidth)
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for \
                  bandwidth_temp in bandwidth_list]

    # return the final kernel matrix
    return sum(kernel_val)


def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    - source : source data
    - target : target data
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


### the fine-tuning model
class CDP_Net(nn.Module):
    def __init__(self, num_classes=256):
        super(CDP_Net, self).__init__()
        # the encoder part
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
        # the fully-connected layer 1
        self.classifier_1 = nn.Sequential(
            nn.Linear(256, 20),
            nn.SELU(),
        )
        # the fully-connected layer 2
        self.classifier_2 = nn.Sequential(
            nn.Linear(20, 20),
            nn.SELU()
        )
        # the fully-connected layer 3
        self.classifier_3 = nn.Sequential(
            nn.Linear(20, 20),
            nn.SELU()
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
        source_2 = self.classifier_2(source_1)
        source_3 = self.classifier_3(source_2)

        # target data flow
        target = self.features(target)
        target = target.view(target.size(0), -1)
        mmd_loss += mmd_rbf(source_0, target)
        target = self.classifier_1(target)
        mmd_loss += mmd_rbf(source_1, target)
        target = self.classifier_2(target)
        mmd_loss += mmd_rbf(source_2, target)

        result = self.final_classifier(source_3)
        return result, mmd_loss


def compute_adjustment_1(Y_profiling, tro, classes=9, eps=1e-12):
    counts = np.bincount(Y_profiling, minlength=classes).astype(np.float64)
    pi = counts / np.maximum(counts.sum(), 1.0)
    adjustments = np.log(np.power(pi, tro) + eps)  # log(pi^tau)
    return adjustments.astype(np.float32)


def calculate_HW(data):
    # 【优化】使用向量化索引代替列表推导式，速度极快
    return HW_byte_np[data.astype(int)]


def CDP_train(epoch, model):

    # ✅ 必须是 train()（否则 BatchNorm 锁死）
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

        if adjust_flag:
            # adjustments shape: [1, C]
            source_preds = source_preds + adjustments

        clf_loss = clf_criterion(source_preds, source_label)



        loss = clf_loss + 0.1 * mmd_loss
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

def addGaussianNoise(traces, noise_level):
    print('Add Gaussian noise...')
    if noise_level == 0:
        return traces
    else:
        output_traces = np.zeros(np.shape(traces))
        print(np.shape(output_traces))
        for trace in range(len(traces)):
            if(trace % 5000 == 0):
                print(str(trace) + '/' + str(len(traces)))
            profile_trace = traces[trace]
            noise = np.random.normal(
                0, noise_level, size=np.shape(profile_trace))
            output_traces[trace] = profile_trace + noise
        return output_traces

def regulateMatrix(M, size):
    #maxlen = max(len(r) for r in M)
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
    real_key_01 = 224  # key of the source domain
    real_key_02 = 224  # key of the target domain
    labeling_method = 'id'
    lambda_ = 0.1  # Penalty coefficient
    preprocess = None  # preprocess method
    batch_size = 200
    total_epoch = 100
    finetune_epoch = 15  # epoch number for fine-tuning
    lr = 0.001  # learning rate
    log_interval = 50  # epoch interval to log training information
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

    # add Gaussian noise to the target domain
    X_attack_target = addGaussianNoise(X_attack_source, noise_level)
    Y_attack_target = np.load(source_file_path + 'Y_attack.npy')

    if labeling_method == 'id':
        class_num = 256
    elif labeling_method == 'hw':
        class_num = 9
        # 【优化】使用向量化函数
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

    if adjust_flag == True :
        flag = "real"
    else:
        flag = "fake"

    model = CDP_Net(num_classes=256)
    pretrained_path = './models/best_pretrained.pth'
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
    start_time = time.time()
    end_time = time.time()
    for epoch in range(1, finetune_epoch + 1):
        print(f'Train Epoch {epoch}:')
        CDP_train(epoch, model)

        with torch.no_grad():  # 【优化】Validation 不需要梯度
            valid_loss = CDP_validation(model)
            if valid_loss < min_loss:
                min_loss = valid_loss
                # 确保保存目录存在
                if not os.path.exists('./models'):
                    os.makedirs('./models')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, './models/' +str(epoch)+str(countermeasure)+ '_'+str(flag)+str(target_finetune_num)+'_best_valid_loss_fine_tuned_cpda_device{}_to_{}.pth'.format(source_device_id, target_device_id))
                print(f'✅ Best model saved at epoch {epoch} with validation loss: {valid_loss:.4f}')
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