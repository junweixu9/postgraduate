import os
from torch.utils.data import Dataset
import torch
import numpy as np
from torch import nn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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
# 2) AES tables / HW map
# =========================
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

InvSbox = [82, 9, 106, 213, 48, 54, 165, 56, 191, 64, 163, 158, 129, 243, 215, 251, 124, 227, 57, 130, 155, 47, 255,
           135, 52, 142, 67, 68, 196, 222, 233, 203, 84, 123, 148, 50, 166, 194, 35, 61, 238, 76, 149, 11, 66, 250, 195,
           78, 8, 46, 161, 102, 40, 217, 36, 178, 118, 91, 162, 73, 109, 139, 209, 37, 114, 248, 246, 100, 134, 104, 152,
           22, 212, 164, 92, 204, 93, 101, 182, 146, 108, 112, 72, 80, 253, 237, 185, 218, 94, 21, 70, 87, 167, 141, 157,
           132, 144, 216, 171, 0, 140, 188, 211, 10, 247, 228, 88, 5, 184, 179, 69, 6, 208, 44, 30, 143, 202, 63, 15, 2,
           193, 175, 189, 3, 1, 19, 138, 107, 58, 145, 17, 65, 79, 103, 220, 234, 151, 242, 207, 206, 240, 180, 230, 115,
           150, 172, 116, 34, 231, 173, 53, 133, 226, 249, 55, 232, 28, 117, 223, 110, 71, 241, 26, 113, 29, 41, 197, 137,
           111, 183, 98, 14, 170, 24, 190, 27, 252, 86, 62, 75, 198, 210, 121, 32, 154, 219, 192, 254, 120, 205, 90, 244,
           31, 221, 168, 51, 136, 7, 199, 49, 177, 18, 16, 89, 39, 128, 236, 95, 96, 81, 127, 169, 25, 181, 74, 13, 45,
           229, 122, 159, 147, 201, 156, 239, 160, 224, 59, 77, 174, 42, 245, 176, 200, 235, 187, 60, 131, 83, 153, 97, 23,
           43, 4, 126, 186, 119, 214, 38, 225, 105, 20, 99, 85, 33, 12, 125]

HW_byte = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2,
           3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2, 3,
           3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3,
           4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4,
           3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5,
           6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4,
           4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5,
           6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8]

Sbox_np = np.array(Sbox, dtype=np.int32)
InvSbox_np = np.array(InvSbox, dtype=np.int32)
HW_byte_np = np.array(HW_byte, dtype=np.int32)


def calculate_HW(data):
    return HW_byte_np[data.astype(int)]


def compute_adjustment_1(Y_profiling, tro, classes=9, eps=1e-12):
    counts = np.bincount(Y_profiling, minlength=classes).astype(np.float64)
    pi = counts / np.maximum(counts.sum(), 1.0)
    adjustments = np.log(np.power(pi, tro) + eps)  # log(pi^tau)
    return adjustments.astype(np.float32)


# =========================
# 3) Model (same as training)
# =========================
class Net(nn.Module):
    def __init__(self, num_classes=9):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=1),
            nn.SELU(),
            nn.BatchNorm1d(8),
            nn.AvgPool1d(kernel_size=2, stride=2),

            nn.Conv1d(8, 16, kernel_size=11),
            nn.SELU(),
            nn.BatchNorm1d(16),
            nn.AvgPool1d(kernel_size=11, stride=11),

            nn.Conv1d(16, 32, kernel_size=2),
            nn.SELU(),
            nn.BatchNorm1d(32),
            nn.AvgPool1d(kernel_size=3, stride=3),
            nn.Flatten()
        )
        self.classifier_1 = nn.Sequential(
            nn.Linear(448, 2),
            nn.SELU(),
        )
        self.final_classifier = nn.Sequential(
            nn.Linear(2, num_classes)
        )

    def forward(self, input):
        x = self.features(input)
        x = x.view(x.size(0), -1)
        x = self.classifier_1(x)
        output = self.final_classifier(x)
        return output


# =========================
# 4) Plots
# =========================
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.ylim((len(classes) - 0.5, -0.5))
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predict label')
    plt.show()


def plot_guessing_entropy(preds, real_key, device_id, model_flag):
    num_averaged = 100
    trace_num_max = 5000

    step = 1
    if 400 < trace_num_max < 1000:
        step = 2
    if 1000 <= trace_num_max < 5000:
        step = 4
    if 5000 <= trace_num_max < 10000:
        step = 5

    num_steps = int(trace_num_max / step)
    guessing_entropy = np.zeros((num_averaged, num_steps))
    success_flag = np.zeros((num_averaged, num_steps))

    if device_id == target_device_id:
        ciphertext = ciphertexts_target
    else:
        ciphertext = ciphertexts_source

    key_guesses = np.arange(256, dtype=np.int32)

    for time in range(num_averaged):
        random_index = np.random.choice(len(ciphertext), trace_num_max, replace=False)
        selected_cipher = ciphertext[random_index]
        selected_preds = preds[random_index]

        c1 = selected_cipher[:, 1].astype(np.int32)
        c5 = selected_cipher[:, 5].astype(np.int32)

        temp = c1[:, np.newaxis] ^ key_guesses[np.newaxis, :]
        initialState = InvSbox_np[temp]
        media_value = initialState ^ c5[:, np.newaxis]

        if labeling_method == 'identity':
            labels = media_value
        else:
            labels = HW_byte_np[media_value]

        row_indices = np.arange(trace_num_max)[:, np.newaxis]
        probs = selected_preds[row_indices, labels]
        log_probs = np.log(probs + 1e-40)

        cumulative_scores = np.cumsum(log_probs, axis=0)
        indices_to_keep = np.arange(0, trace_num_max, step)
        kept_scores = cumulative_scores[indices_to_keep]

        real_key_scores = kept_scores[:, real_key][:, np.newaxis]
        ranks = np.sum(kept_scores > real_key_scores, axis=1)

        guessing_entropy[time, :] = ranks
        success_flag[time, :] = (ranks == 0).astype(int)

    guessing_entropy = np.mean(guessing_entropy, axis=0)
    print(np.argmax(guessing_entropy < 1))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    x = range(0, trace_num_max, step)
    plt.plot(x, guessing_entropy, color='red')
    plt.xlabel('Number of trace')
    plt.ylabel('Guessing entropy')

    plt.subplot(1, 2, 2)
    success_flag = np.sum(success_flag, axis=0)
    success_rate = success_flag / num_averaged
    plt.plot(x, success_rate, color='red')
    plt.xlabel('Number of trace')
    plt.ylabel('Success rate')
    plt.show()


# =========================
# 5) Test/Attack (fix: apply adjustments)
# =========================
def test(model, device_id, disp_GE=True, model_flag='finetuned'):
    model.eval()
    test_loss = 0.0
    correct = 0
    epoch = 0
    clf_criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)

    if device_id == source_device_id:
        test_loader = source_test_loader
        test_num_local = source_test_num
        real_key = real_key_01
        print("[Testing] Source domain")
    else:
        test_loader = target_test_loader
        test_num_local = target_test_num
        real_key = real_key_02
        print("[Testing] Target domain")

    predlist = torch.zeros(0, dtype=torch.long, device='cpu')
    lbllist = torch.zeros(0, dtype=torch.long, device='cpu')
    test_preds_all = torch.zeros((test_num_local, class_num), dtype=torch.float, device='cpu')

    with torch.no_grad():
        for data, label in test_loader:
            if cuda:
                data, label = data.cuda(non_blocking=True), label.cuda(non_blocking=True)

            logits = model(data)

            test_loss += clf_criterion(logits, label).item()

            pred = logits.data.max(1)[1]

            current_batch_size = data.size(0)
            start_idx = epoch * batch_size
            end_idx = start_idx + current_batch_size
            if end_idx <= test_preds_all.shape[0]:
                test_preds_all[start_idx:end_idx, :] = softmax(logits).cpu()

            predlist = torch.cat([predlist, pred.view(-1).cpu()])
            lbllist = torch.cat([lbllist, label.view(-1).cpu()])
            correct += pred.eq(label.data.view_as(pred)).cpu().sum()
            epoch += 1

    test_loss /= len(test_loader)
    print('Test loss: {:.4f}, Test accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    confusion_mat = confusion_matrix(lbllist.numpy(), predlist.numpy())
    plot_confusion_matrix(confusion_mat, classes=range(class_num))

    if disp_GE:
        plot_guessing_entropy(test_preds_all.numpy(), real_key, device_id, model_flag)


# =========================
# 6) Main
# =========================
if __name__ == '__main__':
    source_device_id = 1
    target_device_id = 3

    real_key_01 = 0x21
    real_key_02 = 0x8F

    labeling_method = 'hw'
    preprocess = 'horizontal_standardization'
    batch_size = 200

    train_num = 85000
    valid_num = 5000
    source_test_num = 9900
    target_finetune_num = 200
    target_test_num = 9400

    trace_offset = 0
    trace_length = 1000

    source_file_path = './Data/device1/'
    target_file_path = './Data/device3/'

    no_cuda = False
    cuda = (not no_cuda) and torch.cuda.is_available()

    seed = 8
    torch.manual_seed(seed)
    np.random.seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    # ====== 1) load attack/test data ======
    X_attack_source = np.load(source_file_path + 'X_attack.npy')
    Y_attack_source = np.load(source_file_path + 'Y_attack.npy')
    X_attack_target = np.load(target_file_path + 'X_attack.npy')
    Y_attack_target = np.load(target_file_path + 'Y_attack.npy')

    if labeling_method == 'identity':
        class_num = 256
    else:
        class_num = 9
        Y_attack_source = calculate_HW(Y_attack_source)
        Y_attack_target = calculate_HW(Y_attack_target)

    ciphertexts_source = np.load(source_file_path + 'ciphertexts_attack.npy')
    ciphertexts_target_full = np.load(target_file_path + 'ciphertexts_attack.npy')
    ciphertexts_target = ciphertexts_target_full[target_finetune_num:target_finetune_num + target_test_num]

    # normalize per trace (keep your logic)
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
    kwargs_target_test = {
        'trs_file': X_attack_target[target_finetune_num:target_finetune_num + target_test_num, :],
        'label_file': Y_attack_target[target_finetune_num:target_finetune_num + target_test_num],
        'trace_num': target_test_num,
        'trace_offset': trace_offset,
        'trace_length': trace_length,
    }

    source_test_loader = load_testing(batch_size, kwargs_source_test)
    target_test_loader = load_testing(batch_size, kwargs_target_test)
    print('Load data complete!')

    # ====== 2) (关键) compute adjustments from source TRAIN distribution ======
    # 训练时你用的是 device1 的 Y_train 分布（profiling/train），测试也必须一致
    adjust_flag = True
    adjustments = None
    if adjust_flag:
        # 尝试加载源域训练标签用于计算 pi
        y_train_path = os.path.join(source_file_path, 'Y_train.npy')
        if not os.path.exists(y_train_path):
            raise FileNotFoundError(
                f"adjust_flag=True 但找不到 {y_train_path}，无法计算 logit adjustment。"
            )

        Y_train_source = np.load(y_train_path)
        if labeling_method == 'hw':
            Y_train_source = calculate_HW(Y_train_source)

        adjustments_np = compute_adjustment_1(Y_train_source[:train_num], tro=1, classes=class_num)
        adjustments = torch.from_numpy(adjustments_np).view(1, -1).float()
        if cuda:
            adjustments = adjustments.cuda()

    # ====== 3) load fine-tuned MMD-CORAL model ======
    flag = "real" if adjust_flag else "fake"
    ckpt_path = './models/' + str(flag) + 'best_valid_loss_fine_tuned_mmd_coral_device_GPT_{}_to_{}.pth'.format(
        source_device_id, target_device_id
    )
    print("Loading checkpoint:", ckpt_path)

    model = Net(num_classes=class_num)
    if cuda:
        model.cuda()

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Results after fine-tuning (MMD-CORAL):')

    with torch.no_grad():
        print('Result on source device:')
        test(model, source_device_id, model_flag='finetuned_source')
        print('Result on target device:')
        test(model, target_device_id, model_flag='finetuned_target')
