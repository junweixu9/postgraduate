import os
from torch.utils.data import Dataset
import torch
import numpy as np
from torch import nn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import random

torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch.nn.functional as F

# ==============================
# 1) Dataset / DataLoader
# ==============================
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
        trace_tensor = torch.from_numpy(trace).float().unsqueeze(0)  # (1, L)
        label_tensor = torch.tensor(self.label_file[index], dtype=torch.long)
        return trace_tensor, label_tensor

    def __len__(self):
        return self.trace_num


def load_testing(batch_size, kwargs):
    data = TorchDataset(**kwargs)
    # 【关键修正】测试必须 drop_last=False，否则最后一批会被丢掉 -> preds/plaintext/labels 不对齐
    test_loader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    return test_loader


# ==============================
# 2) AES tables + HW
# ==============================
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


def calculate_HW_np(y):
    return HW_byte_np[y.astype(np.int32)]


def compute_adjustment_1(Y_profiling, tro, classes=9, eps=1e-12):
    counts = np.bincount(Y_profiling, minlength=classes).astype(np.float64)
    pi = counts / np.maximum(counts.sum(), 1.0)
    adjustments = np.log(np.power(pi, tro) + eps)
    return adjustments.astype(np.float32)


# ==============================
# 3) Plot utils
# ==============================
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
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
    preds: (N, C) probabilities (already softmaxed)
    """
    num_averaged = 100
    trace_num_max = min(5000, preds.shape[0])

    if device_id == target_device_id:
        plaintext = plaintexts_target
    else:
        plaintext = plaintexts_source

    # 重要：plaintext 长度必须与 preds 行数一致
    plaintext = plaintext[:preds.shape[0]]

    key_guesses = np.arange(256, dtype=np.int32)
    guessing_entropy = np.zeros((num_averaged, trace_num_max))
    success_flag = np.zeros((num_averaged, trace_num_max))

    for t in range(num_averaged):
        idx = np.random.choice(len(plaintext), trace_num_max, replace=False)
        selected_pt = plaintext[idx]
        selected_preds = preds[idx]

        state = selected_pt[:, np.newaxis] ^ key_guesses[np.newaxis, :]
        sbox_out = Sbox_np[state]

        if labeling_method == 'identity':
            labels = sbox_out
        else:  # hw
            labels = HW_byte_np[sbox_out]

        row = np.arange(trace_num_max)[:, np.newaxis]
        probs = selected_preds[row, labels]
        log_probs = np.log(probs + 1e-40)

        cumulative_scores = np.cumsum(log_probs, axis=0)
        real_key_scores = cumulative_scores[:, real_key][:, np.newaxis]
        ranks = np.sum(cumulative_scores > real_key_scores, axis=1)

        guessing_entropy[t, :] = ranks
        success_flag[t, :] = (ranks == 0).astype(int)

    avg_ge = np.mean(guessing_entropy, axis=0)
    avg_sr = np.mean(success_flag, axis=0)

    converge_idx = np.argmax(avg_ge < 1)
    if avg_ge[converge_idx] >= 1:
        print(f"[{model_flag}] GE did not converge to <1 within {trace_num_max} traces.")
    else:
        print(f"[{model_flag}] Traces to reach GE<1: {converge_idx + 1}")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(avg_ge)
    plt.xlabel('Number of trace')
    plt.ylabel('Guessing entropy')

    plt.subplot(1, 2, 2)
    plt.plot(avg_sr)
    plt.xlabel('Number of trace')
    plt.ylabel('Success rate')
    plt.show()


# ==============================
# 4) Model (same names as training Net/CDP_Net)
# ==============================
class Net(nn.Module):
    def __init__(self, num_classes=9):
        super(Net, self).__init__()
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

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier_1(x)
        x = self.classifier_2(x)
        x = self.classifier_3(x)
        x = self.final_classifier(x)
        return x


# ==============================
# 5) Test function (updated)
# ==============================
def test(model, device_id, model_flag='finetuned', disp_GE=True):
    """
    - 输出：loss/acc + confusion matrix + GE/SR
    - logits adjustment: 与训练保持一致
    """
    model.eval()
    clf_criterion = nn.CrossEntropyLoss()

    if device_id == source_device_id:
        loader = source_test_loader
        real_key = real_key_01
        tag = "Source"
    else:
        loader = target_test_loader
        real_key = real_key_02
        tag = "Target"

    pred_list = []
    lbl_list = []
    prob_list = []

    # eval + no_grad 需要一起用：eval 控制 BN/Dropout，no_grad 关闭 autograd 加速省显存
    # 这两个作用不同。:contentReference[oaicite:2]{index=2}
    with torch.no_grad():
        for data, label in loader:
            if cuda:
                data = data.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)

            logits = model(data)

            # 【关键】如果训练用了 logit adjustment，测试也必须一致地加
            if adjust_flag and (adjustments is not None):
                logits = logits + adjustments  # (C,)

            loss = clf_criterion(logits, label)

            probs = F.softmax(logits, dim=1).detach().cpu()  # (bs,C)
            pred = torch.argmax(probs, dim=1)

            prob_list.append(probs)
            pred_list.append(pred.cpu())
            lbl_list.append(label.detach().cpu())

    probs_all = torch.cat(prob_list, dim=0).numpy()
    preds_all = torch.cat(pred_list, dim=0).numpy()
    labels_all = torch.cat(lbl_list, dim=0).numpy()

    # 重新计算整体 loss / acc（更稳：避免 batch 平均的偏差）
    # 这里只用 acc 为主，loss 用 batch 平均也行
    acc = (preds_all == labels_all).mean() * 100.0
    print(f'{tag} test accuracy: {acc:.2f}%  (N={len(labels_all)})')

    cm = confusion_matrix(labels_all, preds_all)
    plot_confusion_matrix(cm, classes=list(range(class_num)),
                          title=f'{tag} Confusion Matrix - {model_flag}')

    if disp_GE:
        plot_guessing_entropy(probs_all, real_key, device_id, model_flag)


# ==============================
# 6) Clock jitter (your original, kept)
# ==============================
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


# ==============================
# 7) Main
# ==============================
if __name__ == '__main__':
    # ---- config ----
    source_device_id = 0
    target_device_id = 1
    real_key_01 = 224
    real_key_02 = 224

    labeling_method = 'hw'   # 'hw' or 'identity'
    batch_size = 200
    target_finetune_num = 200
    source_test_num = 10000
    target_test_num = 9000
    trace_offset = 0
    trace_length = 700
    countermeasure = '_clockjitter_level1'
    clock_range = 1
    source_file_path = './Data/ASCAD/'

    # logit adjustment
    adjust_flag = True
    tro = 1.0

    no_cuda = False
    cuda = (not no_cuda) and torch.cuda.is_available()
    seed = 8
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    # ---- load attack traces/labels ----
    X_attack_source = np.load(source_file_path + 'X_attack.npy')
    Y_attack_source = np.load(source_file_path + 'Y_attack.npy')

    if labeling_method == 'identity':
        class_num = 256
    else:
        class_num = 9
        Y_attack_source = calculate_HW_np(Y_attack_source)

    # target domain traces
    X_attack_target = addClockJitter(X_attack_source, clock_range, trace_length)
    Y_attack_target = Y_attack_source

    # ---- load plaintexts (align with the SAME slices used by loaders!) ----
    pt_all = np.load(source_file_path + 'plaintexts_attack.npy')  # (N,16)
    pt_byte2 = pt_all[:, 2].astype(np.int32)

    # source test loader uses first source_test_num samples (because trace_num=source_test_num)
    plaintexts_source = pt_byte2[:source_test_num]

    # target test loader uses slice [target_finetune_num : target_finetune_num + target_test_num]
    plaintexts_target = pt_byte2[target_finetune_num: target_finetune_num + target_test_num]

    # ---- loaders ----
    kwargs_source_test = {
        'trs_file': X_attack_source,
        'label_file': Y_attack_source,
        'trace_num': source_test_num,
        'trace_offset': trace_offset,
        'trace_length': trace_length,
    }
    kwargs_target_test = {
        'trs_file': X_attack_target[target_finetune_num: target_finetune_num + target_test_num, :],
        'label_file': Y_attack_target[target_finetune_num: target_finetune_num + target_test_num],
        'trace_num': target_test_num,
        'trace_offset': trace_offset,
        'trace_length': trace_length,
    }

    source_test_loader = load_testing(batch_size, kwargs_source_test)
    target_test_loader = load_testing(batch_size, kwargs_target_test)
    print('Load test data complete!')

    # ---- adjustments: 优先从 checkpoint 读；否则用测试标签分布近似（不完美，但至少一致可跑） ----
    adjustments = None
    if adjust_flag:
        # 这里你如果有 profiling/train 的 labels，建议用它们来算更合理
        # 先用 attack_source 的标签分布作为 fallback
        adj_np = compute_adjustment_1(Y_attack_source[:min(len(Y_attack_source), 50000)], tro, classes=class_num)
        adjustments = torch.from_numpy(adj_np)
        if cuda:
            adjustments = adjustments.cuda(non_blocking=True)

    # ---- load model ----
    model = Net(num_classes=class_num)
    if cuda:
        model.cuda()

    ckpt_path = (
        './models/' + str(adjust_flag) + '_' + str(countermeasure) +
        '_best_valid_loss_fine_tuned_cpda_gpt_device{}_to_{}.pth'.format(source_device_id, target_device_id)
    )
    print("Loading checkpoint:", ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Model loaded.')

    # ---- eval ----
    with torch.no_grad():
        print('\nResult on source device:')
        test(model, source_device_id, model_flag='finetuned_source', disp_GE=True)

        print('\nResult on target device:')
        test(model, target_device_id, model_flag='finetuned_target', disp_GE=True)
