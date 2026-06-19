import os
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from torch import nn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 【加速1】开启 cudnn 自动寻优
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
        # 【加速2】直接转换 Tensor
        trace_tensor = torch.from_numpy(trace).float().unsqueeze(0)
        label_tensor = torch.tensor(self.label_file[index], dtype=torch.long)
        return trace_tensor, label_tensor

    def __len__(self):
        return self.trace_num


### data loader for training
def load_training(batch_size, kwargs):
    data = TorchDataset(**kwargs)
    # 【加速3】多进程 + 持久化进程
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

InvSbox = [82, 9, 106, 213, 48, 54, 165, 56, 191, 64, 163, 158, 129, 243, 215, 251, 124, 227, 57, 130, 155, 47, 255, 135,
           52, 142, 67, 68, 196, 222, 233, 203, 84, 123, 148, 50, 166, 194, 35, 61,238, 76, 149, 11, 66, 250, 195, 78, 8,
           46, 161, 102, 40, 217, 36, 178, 118, 91, 162, 73, 109, 139, 209, 37, 114, 248, 246, 100, 134, 104, 152, 22, 212,
           164, 92, 204, 93, 101, 182, 146, 108, 112, 72, 80, 253, 237, 185, 218, 94, 21, 70, 87, 167, 141, 157, 132, 144,
           216, 171, 0, 140, 188, 211, 10, 247, 228, 88, 5, 184, 179, 69, 6, 208, 44, 30, 143, 202, 63, 15, 2, 193, 175, 189,
           3, 1, 19, 138, 107, 58, 145, 17, 65, 79, 103, 220, 234, 151, 242, 207, 206, 240, 180, 230, 115, 150, 172, 116, 34,
           231, 173, 53, 133, 226, 249, 55, 232, 28, 117, 223, 110, 71, 241, 26, 113, 29, 41, 197, 137, 111, 183, 98, 14, 170,
           24,190, 27, 252, 86, 62, 75, 198, 210, 121, 32, 154, 219, 192, 254, 120, 205, 90, 244, 31, 221, 168, 51, 136, 7,
           199, 49, 177, 18, 16, 89, 39, 128, 236, 95, 96, 81, 127, 169, 25, 181,74, 13, 45, 229, 122, 159, 147, 201, 156,
           239, 160, 224, 59, 77, 174, 42, 245, 176, 200, 235, 187, 60, 131, 83, 153, 97, 23, 43, 4, 126, 186, 119, 214, 38,
           225, 105, 20, 99, 85, 33,12, 125]

HW_byte = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2,
            3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2, 3,
            3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3,
            4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4,
            3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5,
            6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4,
            4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5,
            6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8]

# 【关键】将列表转换为 numpy 数组以进行加速计算
Sbox_np = np.array(Sbox, dtype=np.int32)
HW_byte_np = np.array(HW_byte, dtype=np.int32)

def test(model, device_id, disp_GE=True, model_flag='pretrained'):
    model.eval()
    test_loss = 0
    correct = 0
    epoch = 0
    clf_criterion = nn.CrossEntropyLoss()
    if device_id == source_device_id:
        test_num = source_test_num
        test_loader = source_test_loader
        real_key = real_key_source
    else:
        test_num = target_test_num
        test_loader = target_test_loader
        real_key = real_key_target

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

    confusion_mat = confusion_matrix(lbllist.numpy(), predlist.numpy())
    plot_confusion_matrix(confusion_mat, classes=range(class_num))
    if disp_GE:
        plot_guessing_entropy(test_preds_all.numpy(), real_key, device_id, model_flag)

def plot_guessing_entropy(preds, real_key, device_id, model_flag):
    # GE/SR is averaged over 100 attacks
    num_averaged = 200
    # max trace num for attack
    trace_num_max = 500

    # 初始化结果数组，形状为 (100, 5000)
    guessing_entropy = np.zeros((num_averaged, trace_num_max))
    success_flag = np.zeros((num_averaged, trace_num_max))

    if device_id == target_device_id:
        plaintext = plaintexts_target
    elif device_id == source_device_id:
        plaintext = plaintexts_source

    key_guesses = np.arange(256, dtype=np.int32)

    for time in range(num_averaged):
        # 随机选择 trace
        random_indices = np.random.choice(len(plaintext), trace_num_max, replace=False)
        selected_pt = plaintext[random_indices]
        selected_preds = preds[random_indices]

        # 向量化计算 State: (500, 1) ^ (1, 256) -> (500, 256)
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

        # 累积求和
        cumulative_scores = np.cumsum(log_probs, axis=0)

        # 计算排名
        real_key_scores = cumulative_scores[:, real_key][:, np.newaxis]
        ranks = np.sum(cumulative_scores > real_key_scores, axis=1)

        guessing_entropy[time, :] = ranks
        success_flag[time, :] = (ranks == 0).astype(int)

    # 计算平均 GE
    ge_mean = np.mean(guessing_entropy, axis=0)

    # 【修改点3】结果输出
    # 找到第一个 GE 降为 0 的索引
    first_zero_index = np.argmax(ge_mean == 0 )
    print(first_zero_index)

    # 检查是否真的收敛到了0（防止 argmax 在全非0数组中返回 0）
    if ge_mean[first_zero_index] == 0:
        # 因为索引是从 0 开始的 (index 0 代表使用了 1 条 trace)
        print(f"Attack succeeded at trace: {first_zero_index + 1}")
    else:
        print("Attack failed to converge to GE=0 within max traces.")
    # np.save('./results/success_rate_' + labeling_method + '_{}_to_{}_'.format(source_device_id, device_id) + model_flag, success_rate)


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
            nn.Linear(64, 20),
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

def calculate_HW(data):

    return HW_byte_np[data.astype(int)]

if __name__ == '__main__':

    DEVICE_CONFIG = {
        i: {'key': i, 'folder': f'device{i:02d}'}
        for i in range(1, 9)
    }

    source_device_id = 1  # 源域设备 ID
    target_device_id = 5  # 目标域设备 ID

    if source_device_id not in DEVICE_CONFIG or target_device_id not in DEVICE_CONFIG:
        raise ValueError("设备ID必须在 1-8 之间")

    # 自动获取
    real_key_source = DEVICE_CONFIG[source_device_id]['key']
    real_key_target = DEVICE_CONFIG[target_device_id]['key']

    source_file_path = f"./Data/{DEVICE_CONFIG[source_device_id]['folder']}/"
    target_file_path = f"./Data/{DEVICE_CONFIG[target_device_id]['folder']}/"

    print(f"Source: Device {source_device_id} | Path: {source_file_path} | Key: {hex(real_key_source)}")
    print(f"Target: Device {target_device_id} | Path: {target_file_path} | Key: {hex(real_key_target)}")

    labeling_method = 'hw'
    _lambda = 0.05
    batch_size = 50
    total_epoch = 50
    finetune_epoch = 100
    lr = 0.001
    log_interval = 40
    train_num = 20000
    valid_num = 5000
    source_test_num = 5000
    target_finetune_num = 50
    target_test_num = 4500
    trace_offset = 0
    trace_length = 500
    no_cuda = False
    cuda = not no_cuda and torch.cuda.is_available()
    seed = 8
    torch.manual_seed(seed)

    X_attack_source = np.load(source_file_path + 'X_attack.npy')
    Y_attack_source = np.load(source_file_path + 'Y_attack.npy')
    X_attack_target = np.load(target_file_path + 'X_attack.npy')
    Y_attack_target = np.load(target_file_path + 'Y_attack.npy')

    if cuda:
        torch.cuda.manual_seed(seed)

    if labeling_method == 'identity':
        class_num = 256
    elif labeling_method == 'hw':
        class_num = 9

    plaintexts_source = np.load(source_file_path + 'plaintexts_attack.npy')
    plaintexts_target = np.load(target_file_path + 'plaintexts_attack.npy')
    plaintexts_target = plaintexts_target[target_finetune_num:target_finetune_num + target_test_num]

    mn = np.repeat(np.mean(X_attack_source, axis=1, keepdims=True), X_attack_source.shape[1], axis=1)
    std = np.repeat(np.std(X_attack_source, axis=1, keepdims=True), X_attack_source.shape[1], axis=1)
    X_attack_source = (X_attack_source - mn) / std

    mn = np.repeat(np.mean(X_attack_target, axis=1, keepdims=True), X_attack_target.shape[1], axis=1)
    std = np.repeat(np.std(X_attack_target, axis=1, keepdims=True), X_attack_target.shape[1], axis=1)
    X_attack_target = (X_attack_target - mn) / std

    kwargs_source_test = {
        'trs_file': X_attack_source,
        'label_file': Y_attack_source,
        'trace_num': source_test_num,
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

    source_test_loader = load_testing(batch_size, kwargs_source_test)
    target_test_loader = load_testing(batch_size, kwargs_target)
    print('Load data complete!')

    model = Net(num_classes=9)

    print('Construct model complete')
    if cuda:
        model.cuda()

    adjust_flag = False

    if adjust_flag:
        flag = "real"
    else:
        flag = "fake"

    path = './models/cmmd_real_best_valid_loss_finetuned_device{}_to_{}.pth'.format(source_device_id, target_device_id)

    # Load model if exists
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model_dict = checkpoint['model_state_dict']
        model.load_state_dict(model_dict)
    else:
        print("未找到预训练模型，请先训练或检查路径。")

    # Evaluate
    with torch.no_grad():
        print('Result on source device:')
        test(model, source_device_id, model_flag='pretrained_source')
        print('Result on target device:')
        test(model, target_device_id, model_flag='pretrained_target')