import os
from torch.utils.data import Dataset, DataLoader
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
        x = torch.from_numpy(trace).float().unsqueeze(0)  # [1, L]
        y = torch.tensor(self.label_file[index], dtype=torch.long)
        return x, y

    def __len__(self):
        return self.trace_num


def load_testing(batch_size, kwargs):
    data = TorchDataset(**kwargs)
    # ✅ 测试不要 drop_last=True，否则会丢样本，影响 accuracy/GE
    loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    return loader


# =========================
# 2) Tables (vectorized)
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
           229, 122, 159, 147, 201, 156, 239, 160, 224, 59, 77, 174, 42, 245, 176, 200, 235, 187, 60, 131, 83, 153, 97,
           23, 43, 4, 126, 186, 119, 214, 38, 225, 105, 20, 99, 85, 33, 12, 125]

HW_byte = [bin(n).count("1") for n in range(256)]
Sbox_np = np.array(Sbox, dtype=np.int32)
InvSbox_np = np.array(InvSbox, dtype=np.int32)
HW_byte_np = np.array(HW_byte, dtype=np.int32)


def calculate_HW(data: np.ndarray) -> np.ndarray:
    return HW_byte_np[data.astype(np.int32)]


# =========================
# 3) Model (same as training backbone)
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

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier_1(x)
        return self.final_classifier(x)


# =========================
# 4) Logit adjustment
# =========================
def compute_adjustment_1(Y_profiling, tro, classes=9, eps=1e-12):
    counts = np.bincount(Y_profiling, minlength=classes).astype(np.float64)
    pi = counts / np.maximum(counts.sum(), 1.0)
    # log(pi^tau)
    return (np.log(np.power(pi, tro) + eps)).astype(np.float32)


# =========================
# 5) Plot helpers
# =========================
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.ylim((len(classes) - 0.5, -0.5))
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predict label')
    plt.show()


# =========================
# 6) Guessing Entropy (unchanged logic)
# =========================
def plot_guessing_entropy(preds, real_key, device_id, model_flag):
    num_averaged = 100
    trace_num_max = 5000

    guessing_entropy = np.zeros((num_averaged, trace_num_max))
    success_flag = np.zeros((num_averaged, trace_num_max))

    if device_id == target_device_id:
        ciphertext = ciphertexts_target
    else:
        ciphertext = ciphertexts_source

    key_guesses = np.arange(256, dtype=np.int32)
    row_indices = np.arange(trace_num_max)[:, np.newaxis]

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

        probs = selected_preds[row_indices, labels]
        log_probs = np.log(probs + 1e-40)

        cumulative_scores = np.cumsum(log_probs, axis=0)
        real_key_scores = cumulative_scores[:, real_key][:, np.newaxis]
        ranks = np.sum(cumulative_scores > real_key_scores, axis=1)

        guessing_entropy[time, :] = ranks
        success_flag[time, :] = (ranks == 0).astype(int)

    ge_mean = np.mean(guessing_entropy, axis=0)
    first_zero_index = np.argmax(ge_mean == 0)

    print(first_zero_index)
    if ge_mean[first_zero_index] == 0:
        print(f"Attack succeeded at trace: {first_zero_index + 1}")
    else:
        print("Attack failed to converge to GE=0 within max traces.")


# =========================
# 7) Test (CMMD-trained model is evaluated same way)
# =========================
def test(model, device_id, disp_GE=True, model_flag='finetuned',
         apply_logit_adjust_in_test=False, adjustments=None):
    """
    ✅ 注意：CMMD/MMD 只影响训练（通过对齐损失），测试阶段只需要 forward logits。
    ✅ 若你想做 “post-hoc logit adjustment”，可把 apply_logit_adjust_in_test=True。
    Menon et al. 也明确提到 logit adjustment 可 post-hoc 使用。:contentReference[oaicite:2]{index=2}
    """
    model.eval()
    clf_criterion = nn.CrossEntropyLoss()

    if device_id == source_device_id:
        test_loader = source_test_loader
        real_key = real_key_01
    else:
        test_loader = target_test_loader
        real_key = real_key_02

    # ✅ 精确长度（不丢样本）
    test_num = len(test_loader.dataset)

    predlist = []
    lbllist = []
    test_preds_all = np.zeros((test_num, class_num), dtype=np.float32)

    softmax = nn.Softmax(dim=1)

    total_loss = 0.0
    correct = 0
    ptr = 0

    with torch.no_grad():
        for data, label in test_loader:
            if cuda:
                data = data.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)

            logits = model(data)

            # ✅ 可选：测试阶段也做 post-hoc logit adjustment
            if apply_logit_adjust_in_test and adjustments is not None:
                logits = logits + adjustments

            loss = clf_criterion(logits, label).item()
            total_loss += loss

            pred = logits.argmax(dim=1)
            correct += int((pred == label).sum().item())

            probs = softmax(logits).detach().cpu().numpy()
            bsz = probs.shape[0]
            test_preds_all[ptr:ptr + bsz, :] = probs
            ptr += bsz

            predlist.append(pred.detach().cpu().numpy())
            lbllist.append(label.detach().cpu().numpy())

    total_loss /= max(len(test_loader), 1)
    predlist = np.concatenate(predlist, axis=0)
    lbllist = np.concatenate(lbllist, axis=0)

    acc = 100.0 * correct / test_num
    print(f'Test loss: {total_loss:.4f}, Test accuracy: {correct}/{test_num} ({acc:.2f}%)\n')

    cm = confusion_matrix(lbllist, predlist)
    plot_confusion_matrix(cm, classes=range(class_num), title=f'Confusion Matrix ({model_flag})')

    if disp_GE:
        plot_guessing_entropy(test_preds_all, real_key, device_id, model_flag)


# =========================
# 8) Main
# =========================
if __name__ == '__main__':
    source_device_id = 1
    target_device_id = 2
    real_key_01 = 0x21
    real_key_02 = 0xCD

    labeling_method = 'hw'
    batch_size = 200
    target_finetune_num = 200
    target_test_num = 9400
    source_test_num = 9900
    trace_offset = 0
    trace_length = 1000

    source_file_path = './Data/device1/'
    target_file_path = './Data/device2/'

    no_cuda = False
    cuda = (not no_cuda) and torch.cuda.is_available()

    # -----------------
    # load data (attack)
    # -----------------
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
    ciphertexts_target = np.load(target_file_path + 'ciphertexts_attack.npy')
    ciphertexts_target = ciphertexts_target[target_finetune_num:target_finetune_num + target_test_num]

    # normalize per-trace
    mn = np.repeat(np.mean(X_attack_source, axis=1, keepdims=True), X_attack_source.shape[1], axis=1)
    std = np.repeat(np.std(X_attack_source, axis=1, keepdims=True), X_attack_source.shape[1], axis=1)
    X_attack_source = (X_attack_source - mn) / (std + 1e-12)

    mn = np.repeat(np.mean(X_attack_target, axis=1, keepdims=True), X_attack_target.shape[1], axis=1)
    std = np.repeat(np.std(X_attack_target, axis=1, keepdims=True), X_attack_target.shape[1], axis=1)
    X_attack_target = (X_attack_target - mn) / (std + 1e-12)

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

    # -----------------
    # logit adjustment (need source train labels)
    # -----------------
    # ✅ 为了让“测试阶段也可选加 adjustment”，这里从 Y_train.npy 计算
    # 若你不想测试加，就不用加载这段也行
    adjust_flag = False
    apply_logit_adjust_in_test = False  # ✅ 你想推理也加，就改 True

    adjustments = None
    if adjust_flag:
        Y_train_source = np.load(source_file_path + 'Y_train.npy')
        if labeling_method == 'hw':
            Y_train_source = calculate_HW(Y_train_source)
        adj_np = compute_adjustment_1(Y_train_source, tro=1, classes=class_num)
        adjustments = torch.from_numpy(adj_np).view(1, -1)
        if cuda:
            adjustments = adjustments.cuda(non_blocking=True)

    # -----------------
    # load CMMD-finetuned checkpoint
    # -----------------
    model = Net(num_classes=class_num)
    if cuda:
        model.cuda()
    flag = "real" if adjust_flag else "fake"
    # ✅ 关键：这里换成你“CMMD + logit adjustment”微调保存的文件名
    # 例如你之前保存过：cmmd_*_finetuned_device1_to_2.pth
    FINETUNED_CKPT = f'./models/' + str(flag) + '_best_valid_loss_cmmd_finetuned_device{}_to_{}.pth'.format(source_device_id, target_device_id)
    # 如果你仍沿用原命名，也可以改回：
    # FINETUNED_CKPT = f'./models/realbest_valid_loss_fine_tuned_cpda_device{source_device_id}_to_{target_device_id}.pth'

    print("Loading finetuned model:", FINETUNED_CKPT)
    checkpoint = torch.load(FINETUNED_CKPT, map_location='cuda' if cuda else 'cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    print('Results after CMMD fine-tuning:')
    with torch.no_grad():
        print('Result on source device:')
        test(model, source_device_id,
             model_flag='cmmd_finetuned_source',
             apply_logit_adjust_in_test=apply_logit_adjust_in_test,
             adjustments=adjustments)

        print('Result on target device:')
        test(model, target_device_id,
             model_flag='cmmd_finetuned_target',
             apply_logit_adjust_in_test=apply_logit_adjust_in_test,
             adjustments=adjustments)
