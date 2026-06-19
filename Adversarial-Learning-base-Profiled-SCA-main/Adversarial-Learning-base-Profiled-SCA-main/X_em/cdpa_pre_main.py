import os
import time

from torch.utils.data import Dataset, DataLoader
import torch
from torch import optim
import numpy as np
from torch import nn


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 开启 cudnn 自动寻优
torch.backends.cudnn.benchmark = True

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
        # 直接转换，无需 transforms
        trace_tensor = torch.from_numpy(trace).float().unsqueeze(0)
        label_tensor = torch.tensor(self.label_file[index], dtype=torch.long)
        return trace_tensor, label_tensor

    def __len__(self):
        return self.trace_num


### data loader for training
def load_training(batch_size, kwargs):
    data = TorchDataset(**kwargs)
    # 【核心修改】
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True,
                                               drop_last=True, num_workers=2,
                                               pin_memory=True, persistent_workers=True)
    return train_loader


### data loader for testing
def load_testing(batch_size, kwargs):
    data = TorchDataset(**kwargs)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False,
                                              drop_last=True, num_workers=2,
                                              pin_memory=True, persistent_workers=True)
    return test_loader

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

# 【优化1】将查找表转换为 NumPy 数组，以便进行向量化索引
HW_byte_np = np.array(HW_byte, dtype=np.int32)

def train(epoch, model, freeze_BN = False):
    # Instantiate the iterator

    if freeze_BN:
        model.eval() # enter eval mode to freeze batch normalization layers
    else:
        model.train() # enter training mode
    iter_source = iter(source_train_loader)
    num_iter = len(source_train_loader)
    clf_criterion = nn.CrossEntropyLoss()
    for i in range(1, num_iter + 1):
        source_data, source_label = next(iter_source)
        if cuda:
            source_data = source_data.cuda(non_blocking=True)
            source_label = source_label.cuda(non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        source_preds = model(source_data)
        if adjust_flag:
            source_preds = source_preds + 1 * adjustments
        loss = clf_criterion(source_preds, source_label)
        loss.backward()
        optimizer.step()
        if i % log_interval == 0:
            # 仅在需要打印时计算准确率，节省算力
            preds = source_preds.data.max(1, keepdim=True)[1]
            correct_batch = preds.eq(source_label.data.view_as(preds)).sum()
            print('Train Epoch {}: [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.6f}%'.format(
                epoch, i * len(source_data), len(source_train_loader) * batch_size,
                       100. * i / len(source_train_loader), loss.item(), float(correct_batch) * 100. / batch_size))


def validation(model):
    model.eval()
    valid_loss = 0
    correct_valid = 0
    clf_criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, label in source_valid_loader:
            if cuda:
                data, label = data.cuda(non_blocking=True), label.cuda(non_blocking=True)

            valid_preds = model(data)
            valid_loss += clf_criterion(valid_preds, label).item()
            pred = valid_preds.data.max(1)[1]
            correct_valid += pred.eq(label.data.view_as(pred)).cpu().sum()

    valid_loss /= len(source_valid_loader)
    valid_acc = 100. * correct_valid / len(source_valid_loader.dataset)
    print('Validation: loss: {:.4f}, accuracy: {}/{} ({:.6f}%)'.format(
        valid_loss, correct_valid, len(source_valid_loader.dataset),
        valid_acc))
    return valid_loss, valid_acc


### the pre-trained model
class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
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
    def forward(self, input):
        x = self.features(input)
        x = x.view(x.size(0), -1)
        x = self.classifier_1(x)
        output = self.final_classifier(x)
        return output

def compute_adjustment_1(Y_profiling, tro, classes=9, eps=1e-12):
    """
    Logit Adjustment: add tau * log(pi) to logits (Menon et al. 2020).
    Y_profiling: one-hot 或 soft label，默认取前 classes 维并 argmax 得到类别
    tro: tau
    """
    counts = np.bincount(Y_profiling, minlength=classes).astype(np.float64)
    pi = counts / np.maximum(counts.sum(), 1.0)
    adjustments = np.log(np.power(pi, tro) + eps)  # log(pi^tau)
    return adjustments.astype(np.float32)

def calculate_HW(data):

    return HW_byte_np[data.astype(int)]

if __name__ == '__main__':
    source_device_id = 1
    target_device_id = 2
    real_key_01 = 0x01  # key of the source domain
    real_key_02 = 0x02  # key of the target domain
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
    source_file_path = './Data/device01/'
    target_file_path = './Data/device02/'
    no_cuda = False
    cuda = not no_cuda and torch.cuda.is_available()
    seed = 8
    torch.manual_seed(seed)

    X_train_source = np.load(source_file_path + 'X_train.npy')
    Y_train_source = np.load(source_file_path + 'Y_train.npy')

    if cuda:
        torch.cuda.manual_seed(seed)
    if labeling_method == 'identity':
        class_num = 256
    elif labeling_method == 'hw':
        class_num = 9


    mn = np.repeat(np.mean(X_train_source, axis=1, keepdims=True), X_train_source.shape[1], axis=1)
    std = np.repeat(np.std(X_train_source, axis=1, keepdims=True), X_train_source.shape[1], axis=1)
    X_train_source = (X_train_source - mn) / std

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
    source_train_loader = load_training(batch_size, kwargs_source_train)
    source_valid_loader = load_training(batch_size, kwargs_source_valid)
    print('Load data complete!')

    adjust_flag = False

    if adjust_flag:
        flag = "real"
    else:
        flag = "fake"

    adjustments_np = compute_adjustment_1(Y_train_source[0:train_num], tro=0.5, classes=9)
    adjustments = torch.from_numpy(adjustments_np).view(1, -1)
    if cuda:
        adjustments = adjustments.cuda()

    model = Net(num_classes=class_num)
    print('Construct model complete')
    if cuda:
        model.cuda()
    min_loss = 1000
    optimizer = optim.Adam([
        {'params': model.features.parameters()},
        {'params': model.classifier_1.parameters()},
        {'params': model.final_classifier.parameters()},
    ], lr=lr)
    start_time = time.time()
    end_time = time.time()
    for epoch in range(1, total_epoch + 1):
        print(f'Train Epoch {epoch}:')
        train(epoch, model)

        with torch.no_grad():
            valid_loss, _ = validation(model)
            if valid_loss < min_loss:
                min_loss = valid_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, './models/'+str(flag)+'_pre-trained_cpda_device{}.pth'.format(source_device_id))
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n总训练时间: {elapsed_time:.2f} 秒")
    # 如果需要更友好的格式，可以转换为分钟和秒
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60
    print(f"总训练时间: {minutes} 分 {seconds:.2f} 秒")

    del source_train_loader
    del source_valid_loader


    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    import gc

    gc.collect()

    print("Cleanup complete, exiting...")