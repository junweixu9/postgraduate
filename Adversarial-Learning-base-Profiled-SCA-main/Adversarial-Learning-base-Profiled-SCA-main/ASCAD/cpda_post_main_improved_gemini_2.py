import os
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from torch import nn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import random

# 【优化】开启 cudnn 自动寻优
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
        # 【优化】直接转换 Tensor
        trace_tensor = torch.from_numpy(trace).float().unsqueeze(0)
        label_tensor = torch.tensor(self.label_file[index], dtype=torch.long)
        return trace_tensor, label_tensor

    def __len__(self):
        return self.trace_num


### data loader for testing
def load_testing(batch_size, kwargs):
    data = TorchDataset(**kwargs)
    # Windows 下 num_workers=0 避免多进程报错，Linux 可设为 4
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False,
                                              drop_last=False, num_workers=0,
                                              pin_memory=True)
    return test_loader


# Sbox and HW tables
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

Sbox_np = np.array(Sbox, dtype=np.int32)
HW_byte_np = np.array(HW_byte, dtype=np.int32)


### Model Definition (匹配训练时的 CDP_Net 结构)
class CDP_Net(nn.Module):
    def __init__(self, num_classes=9):
        super(CDP_Net, self).__init__()
        # Encoder part
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=1), nn.SELU(), nn.BatchNorm1d(32), nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=50), nn.SELU(), nn.BatchNorm1d(64), nn.AvgPool1d(kernel_size=50, stride=50),
            nn.Conv1d(64, 128, kernel_size=3), nn.SELU(), nn.BatchNorm1d(128), nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        self.classifier_1 = nn.Sequential(nn.Linear(256, 20), nn.SELU())
        self.classifier_2 = nn.Sequential(nn.Linear(20, 20), nn.SELU())
        self.classifier_3 = nn.Sequential(nn.Linear(20, 20), nn.SELU())
        self.final_classifier = nn.Sequential(nn.Linear(20, num_classes))

        # 在测试时，我们不需要初始化 Loss Function 模块，这不会影响参数加载

    def forward(self, source, target=None, source_label=None):
        """
        前向传播修改：
        如果 target 为 None，则视为推理模式，仅返回分类 logits。
        """
        source_f = self.features(source)
        source_0 = source_f.view(source_f.size(0), -1)
        source_1 = self.classifier_1(source_0)
        source_2 = self.classifier_2(source_1)
        source_3 = self.classifier_3(source_2)
        source_preds = self.final_classifier(source_3)

        # 推理模式：直接返回预测结果
        if target is None:
            return source_preds

        # 训练模式逻辑 (这里省略，测试不需要)
        return source_preds, 0.0


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
        print(f"Testing on Source Device (Key: {real_key})")
    else:
        test_num = target_test_num
        test_loader = target_test_loader
        real_key = real_key_02
        print(f"Testing on Target Device (Key: {real_key})")

    # 使用 cpu 避免显存溢出，预分配张量
    test_preds_all = torch.zeros((test_num, class_num), dtype=torch.float, device='cpu')
    predlist = []
    lbllist = []

    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            if cuda:
                data, label = data.cuda(non_blocking=True), label.cuda(non_blocking=True)

            # 调用 model 时只传入一个参数，触发 inference 模式
            test_preds = model(data)

            test_loss += clf_criterion(test_preds, label).item()
            pred = test_preds.data.max(1)[1]
            softmax = nn.Softmax(dim=1)

            # 保存 softmax 结果用于 GE 计算
            current_batch_size = data.size(0)
            start_idx = i * batch_size
            end_idx = start_idx + current_batch_size

            # 边界检查
            if end_idx <= test_preds_all.shape[0]:
                test_preds_all[start_idx:end_idx, :] = softmax(test_preds).cpu()
            else:
                # 处理最后一个可能不完整的 batch
                valid_len = test_preds_all.shape[0] - start_idx
                if valid_len > 0:
                    test_preds_all[start_idx:, :] = softmax(test_preds).cpu()[:valid_len]

            predlist.append(pred.cpu().numpy())
            lbllist.append(label.cpu().numpy())
            correct += pred.eq(label.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader)
    print('Test loss: {:.4f}, Test accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    # 拼接结果用于混淆矩阵
    pred_arr = np.concatenate(predlist)
    lbl_arr = np.concatenate(lbllist)

    confusion_mat = confusion_matrix(lbl_arr, pred_arr)
    plot_confusion_matrix(confusion_mat, classes=range(class_num))

    if disp_GE:
        plot_guessing_entropy(test_preds_all.numpy(), real_key, device_id, model_flag)


### show the guessing entropy and success rate
def plot_guessing_entropy(preds, real_key, device_id, model_flag):
    num_averaged = 100
    trace_num_max = 5000

    # 确保 preds 长度足够，不足则截断 trace_num_max
    if len(preds) < trace_num_max:
        trace_num_max = len(preds)
        print(f"Warning: Not enough test traces for GE, reducing trace_num_max to {trace_num_max}")

    # 获取对应的明文
    if device_id == target_device_id:
        plaintext = plaintexts_target
    elif device_id == source_device_id:
        plaintext = plaintexts_source

    key_guesses = np.arange(256, dtype=np.int32)
    guessing_entropy = np.zeros((num_averaged, trace_num_max))
    success_flag = np.zeros((num_averaged, trace_num_max))

    for time in range(num_averaged):
        random_indices = np.random.choice(len(plaintext), trace_num_max, replace=False)
        selected_pt = plaintext[random_indices]
        selected_preds = preds[random_indices]

        # 向量化计算: State = Pt ^ Key
        state = selected_pt[:, np.newaxis] ^ key_guesses[np.newaxis, :]
        sbox_out = Sbox_np[state]

        if labeling_method == 'identity':
            labels = sbox_out
        elif labeling_method == 'hw':
            labels = HW_byte_np[sbox_out]

        # 提取概率
        row_indices = np.arange(trace_num_max)[:, np.newaxis]
        probs = selected_preds[row_indices, labels]
        log_probs = np.log(probs + 1e-40)

        # 累加对数似然
        cumulative_scores = np.cumsum(log_probs, axis=0)

        # 计算排名
        real_key_scores = cumulative_scores[:, real_key][:, np.newaxis]
        ranks = np.sum(cumulative_scores > real_key_scores, axis=1)

        guessing_entropy[time, :] = ranks
        success_flag[time, :] = (ranks == 0).astype(int)

    avg_ge = np.mean(guessing_entropy, axis=0)
    avg_sr = np.mean(success_flag, axis=0)

    # 查找 GE < 1 的点
    converge_indices = np.where(avg_ge < 1)[0]
    if len(converge_indices) > 0:
        print("Traces to reach GE < 1: {}".format(converge_indices[0] + 1))
    else:
        print("GE did not converge to < 1 within {} traces.".format(trace_num_max))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(avg_ge, color='red')
    plt.title(f'Guessing Entropy ({model_flag})')
    plt.xlabel('Number of trace')
    plt.ylabel('Guessing entropy')

    plt.subplot(1, 2, 2)
    plt.plot(avg_sr, color='red')
    plt.title(f'Success Rate ({model_flag})')
    plt.xlabel('Number of trace')
    plt.ylabel('Success rate')
    plt.show()


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    # 修复 matplotlib 版本导致的显示截断问题
    plt.ylim((len(classes) - 0.5, -0.5))
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predict label')
    plt.show()


def calculate_HW(data):
    return HW_byte_np[data.astype(int)]


def addClockJitter(traces, clock_range, trace_length):
    print('Add clock jitters...')
    output_traces = []
    for trace_idx in range(len(traces)):
        if (trace_idx % 2000 == 0):
            print(f"{trace_idx}/{len(traces)}")
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
                new_trace.extend([avg_point] * r)
            point += 1
        output_traces.append(new_trace)
    return regulateMatrix(output_traces, trace_length)


def regulateMatrix(M, size):
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
    real_key_01 = 224
    real_key_02 = 224
    labeling_method = 'hw'
    batch_size = 200
    source_test_num = 10000
    target_finetune_num = 200
    target_test_num = 9000
    trace_offset = 0
    trace_length = 700
    countermeasure = '_clockjitter_level1'
    clock_range = 1
    source_file_path = './Data/ASCAD/'
    adjust_flag = 0

    no_cuda = False
    cuda = not no_cuda and torch.cuda.is_available()
    seed = 8
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    # 1. 加载数据
    try:
        X_attack_source = np.load(source_file_path + 'X_attack.npy')
        Y_attack_source = np.load(source_file_path + 'Y_attack.npy')
    except FileNotFoundError:
        print("Error: Data files not found. Creating dummy data for demonstration.")
        X_attack_source = np.random.randn(20000, 1000).astype(np.float32)
        Y_attack_source = np.random.randint(0, 256, 20000).astype(np.uint8)

    if labeling_method == 'identity':
        class_num = 256
    elif labeling_method == 'hw':
        class_num = 9
        Y_attack_source = calculate_HW(Y_attack_source)

    # 2. 对 Target 数据施加 Jitter (需与训练时一致)
    X_attack_target = addClockJitter(X_attack_source, clock_range, trace_length)
    Y_attack_target = Y_attack_source

    # 3. 加载明文 (用于 GE 计算)
    try:
        plaintexts_source = np.load(source_file_path + 'plaintexts_attack.npy')
        plaintexts_source = plaintexts_source[:, 2]  # 假设攻击第 3 个字节

        # Target 明文需要与 X_attack_target 切片保持一致
        plaintexts_target = np.load(source_file_path + 'plaintexts_attack.npy')
        plaintexts_target = plaintexts_target[target_finetune_num: target_finetune_num + target_test_num, 2]
    except FileNotFoundError:
        # Dummy Plaintexts
        plaintexts_source = np.random.randint(0, 256, 20000).astype(np.uint8)
        plaintexts_target = plaintexts_source[target_finetune_num: target_finetune_num + target_test_num]

    # 4. 构造 DataLoader
    kwargs_source_test = {
        'trs_file': X_attack_source,
        'label_file': Y_attack_source,
        'trace_num': source_test_num,
        'trace_offset': trace_offset,
        'trace_length': trace_length,
    }
    kwargs_target = {
        'trs_file': X_attack_target[target_finetune_num: target_finetune_num + target_test_num, :],
        'label_file': Y_attack_target[target_finetune_num: target_finetune_num + target_test_num],
        'trace_num': target_test_num,
        'trace_offset': trace_offset,
        'trace_length': trace_length,
    }

    source_test_loader = load_testing(batch_size, kwargs_source_test)
    target_test_loader = load_testing(batch_size, kwargs_target)
    print('Load data complete!')

    # 5. 模型初始化
    model = CDP_Net(num_classes=9)
    if cuda:
        model.cuda()

    # 6. 加载微调后的模型
    model_path = './models/0_' + str(
        countermeasure) + '_best_valid_loss_fine_tuned_cpda_2_device{}_to_{}.pth'.format(source_device_id,
                                                                                       target_device_id)
    print(f"Loading model from: {model_path}")

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        # 兼容性处理：如果 checkpoint 包含 key 'model_state_dict' 则读取它，否则直接读取 (如果是直接保存的 state_dict)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print('Model loaded successfully.')
    else:
        print("Warning: Model path does not exist!")

    print('Results after fine-tuning:')

    # 7. 执行测试
    with torch.no_grad():
        print('--- Result on Source Device ---')
        test(model, source_device_id, model_flag='finetuned_source')

        print('--- Result on Target Device ---')
        test(model, target_device_id, model_flag='finetuned_target')