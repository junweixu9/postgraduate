import os

import requests
from torch.utils.data import Dataset, DataLoader
import torch
from torch import optim
import numpy as np
import math
from torch import nn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import random
import multiprocessing

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 【优化】开启 cudnn 自动寻优，加速卷积计算
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
        # 直接切片
        trace = self.trs_file[index, self.trace_offset: self.trace_offset + self.trace_length]

        # 【优化】直接转换为 Tensor，避免 ToTensor 和 reshape 的开销
        trace_tensor = torch.from_numpy(trace).float().unsqueeze(0)

        label_tensor = torch.tensor(self.label_file[index], dtype=torch.long)
        return trace_tensor, label_tensor

    def __len__(self):
        return self.trace_num


### data loader for training
def load_training(batch_size, kwargs):
    data = TorchDataset(**kwargs)
    # 【优化】开启多进程 (num_workers=4) 和 persistent_workers
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

# 【优化】转换为 Numpy 数组，供向量化计算使用
Sbox_np = np.array(Sbox, dtype=np.int32)
HW_byte_np = np.array(HW_byte, dtype=np.int32)


### test/attack
def test(model, device_id, disp_GE=True, model_flag='pretrained'):
    model.eval()
    test_loss = 0
    correct = 0
    epoch = 0
    clf_criterion = nn.CrossEntropyLoss()
    if device_id == source_device_id:
        test_num = source_test_num
        test_loader = source_test_loader
        real_key = real_key_01
    else:
        test_num = target_test_num
        test_loader = target_test_loader
        real_key = real_key_02

    # 使用 cpu 避免显存溢出
    predlist = torch.zeros(0, dtype=torch.long, device='cpu')
    lbllist = torch.zeros(0, dtype=torch.long, device='cpu')
    test_preds_all = torch.zeros((test_num, class_num), dtype=torch.float, device='cpu')

    with torch.no_grad():
        for data, label in test_loader:
            if cuda:
                data, label = data.cuda(non_blocking=True), label.cuda(non_blocking=True)

            test_preds = model(data)
            test_loss += clf_criterion(test_preds, label).item()
            pred = test_preds.data.max(1)[1]
            softmax = nn.Softmax(dim=1)

            # 【修正】处理 batch 索引，防止越界 (drop_last=True时通常没问题，但安全起见)
            current_batch_size = data.size(0)
            start_idx = epoch * batch_size
            end_idx = start_idx + current_batch_size

            if end_idx <= test_preds_all.shape[0]:
                test_preds_all[start_idx:end_idx, :] = softmax(test_preds).cpu()

            predlist = torch.cat([predlist, pred.view(-1).cpu()])
            lbllist = torch.cat([lbllist, label.view(-1).cpu()])
            correct += pred.eq(label.data.view_as(pred)).cpu().sum()
            epoch += 1

    test_loss /= len(test_loader)
    print('Target test loss: {:.4f}, Target test accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    # confusion_mat = confusion_matrix(lbllist.numpy(), predlist.numpy())
    # plot_confusion_matrix(confusion_mat, classes=range(class_num))
    if disp_GE:
        plot_guessing_entropy(test_preds_all.numpy(), real_key, device_id, model_flag)


### show the guessing entropy and success rate
def plot_guessing_entropy(preds, real_key, device_id, model_flag):
    """
    【优化版】GE 计算逻辑：
    Input:
    Logic:
        Calculate log likelihood for all key guesses (0-255).
        State = Sbox[Plaintext ^ KeyGuess]
        Label = HW[State] or State
    """
    # GE/SR is averaged over 100 attacks
    num_averaged = 100
    # max trace num for attack
    trace_num_max = 5000

    # 获取对应的明文
    if device_id == target_device_id:
        plaintext = plaintexts_target
    elif device_id == source_device_id:
        plaintext = plaintexts_source

    # 预生成所有可能的密钥猜测 (0-255)
    key_guesses = np.arange(256, dtype=np.int32)

    # 结果容器
    guessing_entropy = np.zeros((num_averaged, trace_num_max))
    success_flag = np.zeros((num_averaged, trace_num_max))

    for time in range(num_averaged):
        # 1. 随机选择 Trace
        # 使用 numpy choice 比 random.shuffle 更快且返回 array
        random_indices = np.random.choice(len(plaintext), trace_num_max, replace=False)
        selected_pt = plaintext[random_indices]  # shape: (5000,)
        selected_preds = preds[random_indices]

        # 2. 向量化计算所有 Traces 对所有 Key 的中间值

        state = selected_pt[:, np.newaxis] ^ key_guesses[np.newaxis, :]

        # 查 Sbox
        sbox_out = Sbox_np[state]

        # 3. 转换为 Label 索引
        if labeling_method == 'identity':
            labels = sbox_out
        elif labeling_method == 'hw':
            labels = HW_byte_np[sbox_out]

        row_indices = np.arange(trace_num_max)[:, np.newaxis]
        probs = selected_preds[row_indices, labels]

        log_probs = np.log(probs + 1e-40)

        # 5. 累加计算 (Cumsum) 模拟 trace 数量增加
        # axis=0 表示沿着 trace 数量方向累加
        cumulative_scores = np.cumsum(log_probs, axis=0)

        # 6. 计算排名
        # 获取正确密钥的分数 (5000, 1)
        real_key_scores = cumulative_scores[:, real_key][:, np.newaxis]

        # 统计有多少个错误的 key 分数 > 正确 key 分数
        ranks = np.sum(cumulative_scores > real_key_scores, axis=1)

        guessing_entropy[time, :] = ranks
        # Rank 0 means 1st place
        success_flag[time, :] = (ranks == 0).astype(int)

    # 计算平均值
    avg_ge = np.mean(guessing_entropy, axis=0)
    avg_sr = np.mean(success_flag, axis=0)

    converge_idx = np.argmax(avg_ge == 0)
    # 如果从未收敛，argmax 返回 0，需要检查一下
    if avg_ge[converge_idx] >= 1:
        print("GE did not converge to < 1 within {} traces.".format(trace_num_max))
    else:
        print("Traces to reach GE < 1: {}".format(converge_idx + 1))



### show the confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.ylim((len(classes) - 0.5, -0.5))
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predict label')
    plt.show()


class Net(nn.Module):
    def __init__(self, num_classes=9):
        super(Net, self).__init__()
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
    def forward(self, input):
        x = self.features(input)
        x = x.view(x.size(0), -1)
        x = self.classifier_1(x)
        x = self.classifier_2(x)
        x = self.classifier_3(x)
        output = self.final_classifier(x)
        return output


def calculate_HW(data):
    # 【优化】向量化计算 HW
    return HW_byte_np[data.astype(int)]


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
    # 将不定长的 list of lists 转换为定长的 numpy array
    # 使用 float32 节省内存并兼容 PyTorch
    Z = np.zeros((len(M), size), dtype=np.float32)
    for enu, row in enumerate(M):
        row_len = len(row)
        if row_len <= size:
            Z[enu, :row_len] = row
        else:
            Z[enu, :] = row[:size]
    return Z


if __name__ == '__main__':
    source_device_id = 0
    target_device_id = 1
    real_key_01 = 224  # key of the source domain
    real_key_02 = 224  # key of the target domain
    labeling_method = 'hw'
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
    target_finetune_num = 200
    target_test_num = 9000
    trace_offset = 0
    trace_length = 700
    countermeasure = '_GaussianNoise_level4'
    noise_level = 4
    source_file_path = './Data/ASCAD/'
    no_cuda = False
    cuda = not no_cuda and torch.cuda.is_available()
    seed = 8
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    # load data
    X_attack_source = np.load(source_file_path + 'X_attack.npy')
    Y_attack_source = np.load(source_file_path + 'Y_attack.npy')
    # add clock jitter to the target domain
    X_attack_target = addGaussianNoise(X_attack_source, noise_level)
    Y_attack_target = np.load(source_file_path + 'Y_attack.npy')

    if labeling_method == 'identity':
        class_num = 256
    elif labeling_method == 'hw':
        class_num = 9
        # 【优化】使用向量化函数
        Y_attack_source = calculate_HW(Y_attack_source)
        Y_attack_target = calculate_HW(Y_attack_source)

    # to load plaintexts
    plaintexts_source = np.load(source_file_path + 'plaintexts_attack.npy')
    plaintexts_source = plaintexts_source[:, 2]

    plaintexts_target = np.load(source_file_path + 'plaintexts_attack.npy')
    plaintexts_target = plaintexts_target[target_finetune_num: target_finetune_num + target_test_num, 2]

    kwargs_source_test = {
        'trs_file': X_attack_source,
        'label_file': Y_attack_source,
        'trace_num': source_test_num,
        'trace_offset': trace_offset,
        'trace_length': trace_length,
    }
    kwargs_target = {
        # 【修正】确保切片范围正确
        'trs_file': X_attack_target[target_finetune_num: target_finetune_num + target_test_num, :],
        'label_file': Y_attack_target[target_finetune_num: target_finetune_num + target_test_num],
        'trace_num': target_test_num,
        'trace_offset': trace_offset,
        'trace_length': trace_length,
    }

    source_test_loader = load_testing(batch_size, kwargs_source_test)
    target_test_loader = load_testing(batch_size, kwargs_target)
    print('Load data complete!')

    model = Net(num_classes=9)
    print('Construct model complete')
    if cuda:
        model.cuda()

    adjust_flag = True
    tro = 1.5
    if adjust_flag == True:

        flag = "real"
        path = './models/hw'+str(countermeasure)+'_'+str(flag)+str(tro)+'_pre-trained_cpda_device{}.pth'.format(source_device_id)
    else:
        flag = "fake"
        path = './models/hw'+str(countermeasure)+'_'+str(flag)+'_pre-trained_cpda_device{}.pth'.format(source_device_id)

    print("Loading model:", path)

    # Load model if exists
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model_dict = checkpoint['model_state_dict']
        model.load_state_dict(model_dict)
    else:
        print(f"Warning: Pretrained model not found at {path}. Results will be random guesses.")

    # Evaluate
    with torch.no_grad():
        print('Result on source device:')
        test(model, source_device_id, model_flag='pretrained_source')
        print('Result on target device:')
        test(model, target_device_id, model_flag='pretrained_target')
    ret = requests.get('https://api.day.app/Cqdu3932Dcbe3dduH2EJPM/训练/完成')