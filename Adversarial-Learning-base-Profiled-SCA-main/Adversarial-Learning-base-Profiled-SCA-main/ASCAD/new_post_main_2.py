import os
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from torch import nn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
torch.backends.cudnn.benchmark = True
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# =========================
# 1) Dataset / DataLoader
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
        data, batch_size=batch_size, shuffle=True,
        drop_last=True, num_workers=4,
        pin_memory=True, persistent_workers=True
    )
    return train_loader


def load_testing(batch_size, kwargs):
    data = TorchDataset(**kwargs)
    test_loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=False,
        drop_last=True, num_workers=4,
        pin_memory=True, persistent_workers=True
    )
    return test_loader


# =========================
# 2) (kept) logit adjustment helper
#    注意：本测试脚本不会用它
# =========================
def compute_adjustment_1(Y_profiling, tro, classes=9, eps=1e-12):
    counts = np.bincount(Y_profiling, minlength=classes).astype(np.float64)
    pi = counts / np.maximum(counts.sum(), 1.0)
    adjustments = np.log(np.power(pi, tro) + eps)  # log(pi^tau)
    return adjustments.astype(np.float32)


# =========================
# 3) AES tables (kept)
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


# =========================
# 4) Evaluation helpers
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
    """
    preds: (N_traces, 9) or (N_traces, 256) probabilities (softmax outputs)
    """
    num_averaged = 100
    trace_num_max = 5000

    if device_id == target_device_id:
        plaintext = plaintexts_target
    else:
        plaintext = plaintexts_source

    key_guesses = np.arange(256, dtype=np.int32)

    guessing_entropy = np.zeros((num_averaged, trace_num_max))
    success_flag = np.zeros((num_averaged, trace_num_max))

    for time in range(num_averaged):
        random_indices = np.random.choice(len(plaintext), trace_num_max, replace=False)
        selected_pt = plaintext[random_indices]
        selected_preds = preds[random_indices]

        state = selected_pt[:, np.newaxis] ^ key_guesses[np.newaxis, :]
        sbox_out = Sbox_np[state]

        if labeling_method == 'identity':
            labels = sbox_out
        else:
            labels = HW_byte_np[sbox_out]

        row_indices = np.arange(trace_num_max)[:, np.newaxis]
        probs = selected_preds[row_indices, labels]
        log_probs = np.log(probs + 1e-40)

        cumulative_scores = np.cumsum(log_probs, axis=0)
        real_key_scores = cumulative_scores[:, real_key][:, np.newaxis]
        ranks = np.sum(cumulative_scores > real_key_scores, axis=1)

        guessing_entropy[time, :] = ranks
        success_flag[time, :] = (ranks == 0).astype(int)

    avg_ge = np.mean(guessing_entropy, axis=0)
    avg_sr = np.mean(success_flag, axis=0)

    converge_idx = np.argmax(avg_ge < 1)
    if avg_ge[converge_idx] >= 1:
        print(f"[{model_flag}] GE did not converge to < 1 within {trace_num_max} traces.")
    else:
        print(f"[{model_flag}] Traces to reach GE < 1: {converge_idx + 1}")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(avg_ge, color='red')
    plt.xlabel('Number of trace')
    plt.ylabel('Guessing entropy')

    plt.subplot(1, 2, 2)
    plt.plot(avg_sr, color='red')
    plt.xlabel('Number of trace')
    plt.ylabel('Success rate')
    plt.show()


def test(model, device_id, disp_GE=True, model_flag='finetuned'):
    """
    注意：测试期间不做 logit adjustment（你要求）
    """
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

    predlist = torch.zeros(0, dtype=torch.long, device='cpu')
    lbllist = torch.zeros(0, dtype=torch.long, device='cpu')
    test_preds_all = torch.zeros((test_num, class_num), dtype=torch.float, device='cpu')

    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for data, label in test_loader:
            if cuda:
                data, label = data.cuda(non_blocking=True), label.cuda(non_blocking=True)

            logits = model(data)  # deep logits only
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
# 5) MKWJDAN-compatible model
# =========================
class MKWJDAN_Net(nn.Module):
    """
    结构基于你原 Net：
    - features -> classifier_1/2/3 -> final_classifier (deep logits)
    - 新增 shallow_head（兼容你训练时加的 shallow_head）
    测试时 forward() 仅输出 deep logits；shallow_head 不参与推理
    """
    def __init__(self, num_classes=9):
        super(MKWJDAN_Net, self).__init__()
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

        self.classifier_1 = nn.Sequential(nn.Linear(256, 20), nn.SELU())
        self.classifier_2 = nn.Sequential(nn.Linear(20, 20), nn.SELU())
        self.classifier_3 = nn.Sequential(nn.Linear(20, 20), nn.SELU())
        self.final_classifier = nn.Sequential(nn.Linear(20, num_classes))

        # 新增：shallow head（用于训练阶段的 JS rectification；测试不必用）
        self.shallow_head = nn.Sequential(
            nn.Linear(256, 20),
            nn.SELU(),
            nn.Linear(20, num_classes)
        )

    def forward(self, input):
        x = self.features(input)           # (B,256)
        x = x.view(x.size(0), -1)
        x = self.classifier_1(x)
        x = self.classifier_2(x)
        x = self.classifier_3(x)
        output = self.final_classifier(x)  # deep logits
        return output

    # 如果你想调试 shallow logits，也可以用这个（测试默认不用）
    def forward_shallow(self, input):
        feat = self.features(input)
        feat = feat.view(feat.size(0), -1)
        return self.shallow_head(feat)


# =========================
# 6) HW + clock jitter (kept)
# =========================
def calculate_HW(data):
    return HW_byte_np[data.astype(int)]


def addClockJitter(traces, clock_range, trace_length):
    print('Add clock jitters...')
    output_traces = []
    for trace_idx in range(len(traces)):
        if trace_idx % 2000 == 0:
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


# =========================
# 7) Main
# =========================
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

    no_cuda = False
    cuda = (not no_cuda) and torch.cuda.is_available()
    seed = 8
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    # ---- load data (attack set only, consistent with your test script) ----
    X_attack_source = np.load(source_file_path + 'X_attack.npy')
    Y_attack_source = np.load(source_file_path + 'Y_attack.npy')

    if labeling_method == 'identity':
        class_num = 256
    else:
        class_num = 9
        Y_attack_source = calculate_HW(Y_attack_source)

    # add clock jitter to target domain
    X_attack_target = addClockJitter(X_attack_source, clock_range, trace_length)
    Y_attack_target = Y_attack_source

    # plaintexts (byte index 2)
    plaintexts_source = np.load(source_file_path + 'plaintexts_attack.npy')[:, 2]
    plaintexts_target_all = np.load(source_file_path + 'plaintexts_attack.npy')
    plaintexts_target = plaintexts_target_all[target_finetune_num: target_finetune_num + target_test_num, 2]

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

    # ---- construct model (MKWJDAN compatible) ----
    model = MKWJDAN_Net(num_classes=class_num)
    print('Construct model complete')
    if cuda:
        model.cuda()

    # 仅用于拼文件名（测试期间不会用 logit adjustment）
    adjust_flag = True
    flag = "real" if adjust_flag else "fake"

    # ---- load checkpoint ----
    # 你需要把这里的文件名改成你训练时实际保存的那个：
    # 我建议你把“mmsd”换成“mkwjdan”（或你实际前缀）
    ckpt_path ='./models/new_clockjitter_level1_real_best_valid_loss_fine_tuned_mkwjdan_device0_to_1.pth'

    print("Loading checkpoint:", ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    finetuned_dict = checkpoint.get('model_state_dict', checkpoint)

    # strict=False：允许 shallow_head 在 ckpt 里有/没有都能加载
    msg = model.load_state_dict(finetuned_dict, strict=False)
    print("Missing keys:", msg.missing_keys)
    print("Unexpected keys:", msg.unexpected_keys)

    print('Results after fine-tuning (NO logit adjustment in test):')

    with torch.no_grad():
        print('Result on source device:')
        test(model, source_device_id, model_flag='finetuned_source_mkwjdan')

        print('Result on target device:')
        test(model, target_device_id, model_flag='finetuned_target_mkwjdan')
