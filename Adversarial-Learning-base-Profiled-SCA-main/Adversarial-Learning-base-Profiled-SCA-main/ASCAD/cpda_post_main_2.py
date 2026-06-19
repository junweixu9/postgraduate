import os
import time

from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from torch import nn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
torch.backends.cudnn.benchmark = True
import random
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


def compute_adjustment_1(Y_profiling, tro, classes=9, eps=1e-12):
    counts = np.bincount(Y_profiling, minlength=classes).astype(np.float64)
    pi = counts / np.maximum(counts.sum(), 1.0)
    adjustments = np.log(np.power(pi, tro) + eps)  # log(pi^tau)
    return adjustments.astype(np.float32)


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
    # GE/SR is averaged over 100 attacks
    num_averaged = 100
    # max trace num for attack
    trace_num_max = 5000

    # 【新增】定义要攻击的字节索引 (例如攻击第0个字节)
    # 如果你的 real_key 是单字节整数，这里对应 plaintext 的哪一列
    target_byte = 0

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

    # 预先生成行索引 (5000, 1)
    row_indices = np.arange(trace_num_max)[:, np.newaxis]

    for time in range(num_averaged):
        # 1. 随机选择 Trace
        random_indices = np.random.choice(len(plaintext), trace_num_max, replace=False)

        # 【修改点1】提取特定字节并转为 int32
        # 如果 plaintext 是 (N, 16)，我们需要取 (N, )
        # 如果 plaintext 本身就是 (N, )，则不需要 [:, target_byte]
        if plaintext.ndim > 1:
            selected_pt = plaintext[random_indices, target_byte].astype(np.int32)
        else:
            selected_pt = plaintext[random_indices].astype(np.int32)

        selected_preds = preds[random_indices]  # shape: (5000, 256) 或 (5000, 9)

        # 2. 向量化计算 State
        # selected_pt: (5000,) -> (5000, 1)
        # key_guesses: (256,) -> (1, 256)
        # state: (5000, 256)
        state = selected_pt[:, np.newaxis] ^ key_guesses[np.newaxis, :]

        # 查 Sbox
        sbox_out = Sbox_np[state]  # (5000, 256)

        # 3. 转换为 Label 索引
        if labeling_method == 'identity':
            labels = sbox_out
        elif labeling_method == 'hw':
            labels = HW_byte_np[sbox_out]

        # 4. 提取概率
        # selected_preds[row, col]
        probs = selected_preds[row_indices, labels]  # (5000, 256)

        # 计算 Log 似然
        log_probs = np.log(probs + 1e-40)

        # 5. 累加计算 (Cumsum)
        # axis=0 沿着 trace 增加的方向累加
        cumulative_scores = np.cumsum(log_probs, axis=0)  # (5000, 256)

        # 6. 计算排名
        # 获取正确密钥的分数 (5000, 1)
        real_key_scores = cumulative_scores[:, real_key][:, np.newaxis]

        # 统计每一条 Trace 处，有多少个错误的 key 分数 > 正确 key 分数
        ranks = np.sum(cumulative_scores > real_key_scores, axis=1)

        guessing_entropy[time, :] = ranks
        success_flag[time, :] = (ranks == 0).astype(int)

    # 计算 100 次攻击的平均排名
    ge_mean = np.mean(guessing_entropy, axis=0)

    # 【修改点2】结果输出逻辑
    # 找到第一个 GE 降为 0 的索引
    first_zero_index = np.argmax(ge_mean == 0)
    print(first_zero_index)
    first_zero_index = np.argmax(ge_mean <1)
    print(first_zero_index)

    # 绘图
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    p1, = plt.plot(ge_mean, color='red')
    plt.xlabel('Number of trace')
    plt.ylabel('Guessing entropy')
    plt.show()

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

def addClockJitter(traces, clock_range, trace_length):
    print('Add clock jitters...')
    output_traces = []
    # 使用 numpy array 预分配，虽然内部逻辑必须循环，但外层可以优化
    # 由于 jitter 后的长度不确定，这里保持 list append，最后统一转换
    for trace_idx in range(len(traces)):
        if (trace_idx % 2000 == 0):
            print(f"{trace_idx}/{len(traces)}")

        trace = traces[trace_idx]
        point = 0
        new_trace = []

        # 优化：提前获取 random 值以减少循环内的调用开销 (Optional, keep logic simple here)
        while point < len(trace) - 1:
            new_trace.append(int(trace[point]))
            r = random.randint(-clock_range, clock_range)
            if r <= 0:
                point += abs(r)
            else:
                # 插入插值
                avg_point = int((int(trace[point]) + int(trace[point + 1])) / 2)
                new_trace.extend([avg_point] * r)  # 使用 extend 替代循环 append
            point += 1
        output_traces.append(new_trace)
    return regulateMatrix(output_traces, trace_length)


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
    finetune_epoch = 30  # epoch number for fine-tuning
    lr = 0.001  # learning rate
    log_interval = 50  # epoch interval to log training information
    train_num = 45000
    valid_num = 5000
    source_test_num = 10000
    target_finetune_num = batch_size
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

    # load data
    X_attack_source = np.load(source_file_path + 'X_attack.npy')
    Y_attack_source = np.load(source_file_path + 'Y_attack.npy')

    if labeling_method == 'identity':
        class_num = 256
    elif labeling_method == 'hw':
        class_num = 9
        # 【优化】使用向量化函数
        Y_attack_source = calculate_HW(Y_attack_source)

    # add clock jitter to the target domain
    X_attack_target = addClockJitter(X_attack_source, clock_range, trace_length)
    Y_attack_target = Y_attack_source

    # to load plaintexts
    plaintexts_source = np.load(source_file_path + 'plaintexts_attack.npy')
    # 假设 plaintexts shape 是 (N, 16)，取第 2 个字节
    plaintexts_source = plaintexts_source[:, 2]

    plaintexts_target = np.load(source_file_path + 'plaintexts_attack.npy')
    # 【注意】这里切片需要对应 X_attack_target 的切片逻辑
    # 下方 kwargs_target 取了 [finetune : finetune+test]，所以这里也应该对应
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

    # create a network
    model = Net(num_classes=9)
    print('Construct model complete')
    if cuda:
        model.cuda()

    adjust_flag = True

    if adjust_flag == True:
        flag = "real"
    else:
        flag = "fake"

    # load the fine-tuned model
    checkpoint = torch.load(
        './models/' +str(countermeasure)+'_'+str(flag)+str(target_finetune_num)+'_best_valid_loss_fine_tuned_cpda_device{}_to_{}.pth'.format(source_device_id, target_device_id))
    finetuned_dict = checkpoint['model_state_dict']
    model.load_state_dict(finetuned_dict)
    print('Results after fine-tuning:')
    # evaluate the fine-tuned model on source and target domain
    with torch.no_grad():
        # print('Result on source device:')
        # test(model, source_device_id, model_flag='finetuned_source')
        start_time = time.time()

        print('Result on target device:')
        test(model, target_device_id, model_flag='finetuned_target')

        end_time = time.time()

        elapsed_time = end_time - start_time
        print(f"\n总训练时间: {elapsed_time:.2f} 秒")
        # 如果需要更友好的格式，可以转换为分钟和秒
        minutes = int(elapsed_time // 60)
        seconds = elapsed_time % 60
        print(f"总训练时间: {minutes} 分 {seconds:.2f} 秒")