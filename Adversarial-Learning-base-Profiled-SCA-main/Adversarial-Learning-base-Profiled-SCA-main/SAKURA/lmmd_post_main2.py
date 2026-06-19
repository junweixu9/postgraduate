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
# Dataset & Loader
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


def load_testing(batch_size, kwargs):
    data = TorchDataset(**kwargs)
    test_loader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    return test_loader


# =========================
# AES tables
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
           135,
           52, 142, 67, 68, 196, 222, 233, 203, 84, 123, 148, 50, 166, 194, 35, 61, 238, 76, 149, 11, 66, 250, 195, 78,
           8, 46, 161, 102, 40, 217, 36, 178, 118, 91, 162, 73, 109, 139, 209, 37, 114, 248, 246, 100, 134, 104, 152,
           22,
           212, 164, 92, 204, 93, 101, 182, 146, 108, 112, 72, 80, 253, 237, 185, 218, 94, 21, 70, 87, 167, 141, 157,
           132,
           144, 216, 171, 0, 140, 188, 211, 10, 247, 228, 88, 5, 184, 179, 69, 6, 208, 44, 30, 143, 202, 63, 15, 2, 193,
           175,
           189, 3, 1, 19, 138, 107, 58, 145, 17, 65, 79, 103, 220, 234, 151, 242, 207, 206, 240, 180, 230, 115, 150,
           172,
           116, 34, 231, 173, 53, 133, 226, 249, 55, 232, 28, 117, 223, 110, 71, 241, 26, 113, 29, 41, 197, 137, 111,
           183, 98,
           14, 170, 24, 190, 27, 252, 86, 62, 75, 198, 210, 121, 32, 154, 219, 192, 254, 120, 205, 90, 244, 31, 221,
           168, 51,
           136, 7, 199, 49, 177, 18, 16, 89, 39, 128, 236, 95, 96, 81, 127, 169, 25, 181, 74, 13, 45, 229, 122, 159,
           147, 201,
           156, 239, 160, 224, 59, 77, 174, 42, 245, 176, 200, 235, 187, 60, 131, 83, 153, 97, 23, 43, 4, 126, 186, 119,
           214,
           38, 225, 105, 20, 99, 85, 33, 12, 125]

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


# =========================
# ✅ Model (与训练代码保持一致的结构)
# =========================
class CDP_Net(nn.Module):
    """
    ✅ 必须与训练代码保持一致，包括 LMMD_loss 组件
    即使测试时不使用 LMMD，也需要保留结构以正确加载 checkpoint
    """

    def __init__(self, num_classes=256):
        super(CDP_Net, self).__init__()
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

        # ✅ 必须保留这个组件（即使测试时不用）
        # 注意：这里为了简化，不初始化 LMMD_loss，因为只需要结构匹配
        # 如果需要完整结构，请复制 LMMD_loss 类定义

    def forward(self, input):
        """测试时的 forward（不需要目标域）"""
        x = self.features(input)
        x = x.view(x.size(0), -1)
        x = self.classifier_1(x)
        return self.final_classifier(x)


# =========================
# ✅ Checkpoint loader（改进版）
# =========================
def load_checkpoint_safely(model: nn.Module, ckpt_path: str, map_location: str = "cpu") -> None:
    """
    安全加载 checkpoint，兼容多种保存格式
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    obj = torch.load(ckpt_path, map_location=map_location)

    if isinstance(obj, dict) and "model_state_dict" in obj:
        # ✅ 使用 strict=False 允许部分 key 不匹配（如 lmmd_loss）
        model.load_state_dict(obj["model_state_dict"], strict=False)
        print("✅ Loaded model_state_dict from checkpoint")
    elif isinstance(obj, dict):
        model.load_state_dict(obj, strict=False)
        print("✅ Loaded state_dict from checkpoint")
    elif isinstance(obj, nn.Module):
        model.load_state_dict(obj.state_dict(), strict=False)
        print("✅ Loaded from saved model object")
    else:
        raise RuntimeError(f"Unrecognized checkpoint format: {type(obj)}")


# =========================
# Plot utils
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


def plot_guessing_entropy(preds, real_key, device_id, model_flag):
    num_averaged = 100
    trace_num_max = 5000

    step = 1
    if 400 < trace_num_max < 1000:
        step = 2
    elif 1000 <= trace_num_max < 5000:
        step = 4
    elif 5000 <= trace_num_max < 10000:
        step = 5

    num_steps = int(trace_num_max / step)
    guessing_entropy = np.zeros((num_averaged, num_steps))
    success_flag = np.zeros((num_averaged, num_steps))

    ciphertext = ciphertexts_target if device_id == target_device_id else ciphertexts_source
    key_guesses = np.arange(256, dtype=np.int32)

    for t in range(num_averaged):
        ridx = np.random.choice(len(ciphertext), trace_num_max, replace=False)
        selected_cipher = ciphertext[ridx]
        selected_preds = preds[ridx]

        c1 = selected_cipher[:, 1].astype(np.int32)
        c5 = selected_cipher[:, 5].astype(np.int32)

        temp = c1[:, None] ^ key_guesses[None, :]
        initialState = InvSbox_np[temp]
        media_value = initialState ^ c5[:, None]

        if labeling_method == 'identity':
            labels = media_value
        else:
            labels = HW_byte_np[media_value]

        row_idx = np.arange(trace_num_max)[:, None]
        probs = selected_preds[row_idx, labels]
        log_probs = np.log(probs + 1e-40)

        cum_scores = np.cumsum(log_probs, axis=0)
        keep = np.arange(0, trace_num_max, step)
        kept_scores = cum_scores[keep]

        real_scores = kept_scores[:, real_key][:, None]
        ranks = np.sum(kept_scores > real_scores, axis=1)

        guessing_entropy[t, :] = ranks
        success_flag[t, :] = (ranks == 0).astype(int)

    avg_ge = np.mean(guessing_entropy, axis=0)
    avg_sr = np.mean(success_flag, axis=0)

    print("Traces to reach GE < 1 (approx index):", int(np.argmax(avg_ge < 1)))

    plt.figure(figsize=(12, 4))
    x = range(0, trace_num_max, step)

    plt.subplot(1, 2, 1)
    plt.plot(x, avg_ge, color='red')
    plt.xlabel('Number of trace')
    plt.ylabel('Guessing entropy')

    plt.subplot(1, 2, 2)
    plt.plot(x, avg_sr, color='red')
    plt.xlabel('Number of trace')
    plt.ylabel('Success rate')
    plt.show()


# =========================
# Test
# =========================
def test(model, device_id, disp_GE=True, model_flag='finetuned'):
    model.eval()
    test_loss = 0.0
    correct = 0
    epoch = 0
    clf_criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)

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

    with torch.no_grad():
        for data, label in test_loader:
            if cuda:
                data = data.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)

            logits = model(data)
            test_loss += clf_criterion(logits, label).item()
            pred = logits.data.max(1)[1]

            bs = data.size(0)
            s = epoch * batch_size
            e = s + bs
            if e <= test_preds_all.shape[0]:
                test_preds_all[s:e, :] = softmax(logits).cpu()

            predlist = torch.cat([predlist, pred.view(-1).cpu()])
            lbllist = torch.cat([lbllist, label.view(-1).cpu()])
            correct += pred.eq(label.data.view_as(pred)).cpu().sum()
            epoch += 1

    test_loss /= len(test_loader)
    acc = 100.0 * correct / len(test_loader.dataset)
    print(f'Test loss: {test_loss:.4f}, accuracy: {correct}/{len(test_loader.dataset)} ({acc:.2f}%)\n')

    cm = confusion_matrix(lbllist.numpy(), predlist.numpy())
    plot_confusion_matrix(cm, classes=range(class_num))

    if disp_GE:
        plot_guessing_entropy(test_preds_all.numpy(), real_key, device_id, model_flag)


# =========================
# Main
# =========================
if __name__ == '__main__':
    source_device_id = 1
    target_device_id = 2
    real_key_01 = 0x21
    real_key_02 = 0xCD

    labeling_method = 'identity'
    batch_size = 200
    source_test_num = 9900
    target_finetune_num = 200
    target_test_num = 9400
    trace_offset = 0
    trace_length = 1000

    source_file_path = './Data/device1/'
    target_file_path = './Data/device2/'

    no_cuda = False
    cuda = (not no_cuda) and torch.cuda.is_available()

    seed = 8
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    # ===== Load data =====
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

    # ===== Standardization =====
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

    # ===== ✅ 修正：明确指定要加载的模型类型 =====
    # 如果训练时使用了 logit adjustment，设为 True
    # 如果训练时未使用 logit adjustment，设为 False
    use_adjusted_model = False  # ✅ 根据实际训练情况设置
    flag = "real" if use_adjusted_model else "fake"

    # ===== Build model =====
    model = CDP_Net(num_classes=class_num)
    if cuda:
        model.cuda()

    # ===== Load checkpoint =====
    ckpt_path = './models/IDlmmd' + str(flag) + '_best_valid_loss_fine_tuned_cpda_device{}_to_{}.pth'.format(
        source_device_id, target_device_id
    )
    print("Loading checkpoint:", ckpt_path)

    try:
        load_checkpoint_safely(model, ckpt_path, map_location="cuda" if cuda else "cpu")
        print(f"✅ Successfully loaded: {ckpt_path}")
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        exit(1)

    # ===== Test =====
    print('\n' + '=' * 60)
    print('Testing fine-tuned model (NO logit adjustment in inference)')
    print('=' * 60 + '\n')

    with torch.no_grad():
        print('📊 Result on source device:')
        test(model, source_device_id, disp_GE=True, model_flag='finetuned_source')

        print('\n' + '-' * 60 + '\n')

        print('📊 Result on target device:')
        test(model, target_device_id, disp_GE=True, model_flag='finetuned_target')

    print('\n' + '=' * 60)
    print('✅ Testing completed!')
    print('=' * 60)