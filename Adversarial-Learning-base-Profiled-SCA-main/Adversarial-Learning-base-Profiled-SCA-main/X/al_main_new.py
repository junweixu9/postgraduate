import os
import time

# 移除 transforms，直接用 torch 处理
from torch.utils.data import Dataset, DataLoader
import torch
from torch import optim
# Variable 已废弃
import numpy as np
import math
from torch import nn
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import matplotlib.pyplot as plt
import itertools
import random
import multiprocessing  # 用于自动检测 CPU 核心数

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 【优化】开启 cudnn 自动寻优
torch.backends.cudnn.benchmark = True


### handle the dataset
class TorchDataset(Dataset):
    def __init__(self, trs_file, label_file, trace_num, trace_offset, trace_length):
        self.trs_file = trs_file
        self.label_file = label_file
        self.trace_num = trace_num
        self.trace_offset = trace_offset
        self.trace_length = trace_length
        # 移除了 ToTensor

    def __getitem__(self, i):
        index = i % self.trace_num
        # 直接切片获取 numpy 数组
        trace = self.trs_file[index, self.trace_offset: self.trace_offset + self.trace_length]

        # 【优化】直接转换为 Tensor 并增加通道维度 (C, L) -> (1, 500)
        # 替代原代码中 np.reshape -> transforms -> np.reshape
        trace_tensor = torch.from_numpy(trace).float().unsqueeze(0)

        label_tensor = torch.tensor(self.label_file[index], dtype=torch.long)
        return trace_tensor, label_tensor

    def __len__(self):
        return self.trace_num


### data loader for training
def load_training(batch_size, kwargs):
    data = TorchDataset(**kwargs)
    # 【优化】开启多进程和持久化 worker
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
           3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5,
           4, 5, 5,
           6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4,
           4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5,
           6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8]


def ALPA_train(epoch, atn_model, critic):
    """
    - epoch : the current epoch
    - atn_model: the adversarial transfer network
    - critic: the Discriminator
    """
    # enter training mode
    atn_model.train()
    critic.train()

    # Instantiate the Iterator for source profiling traces
    iter_source = iter(source_train_loader)

    # 【优化】预加载所有 Target 数据并直接移动到 GPU
    # 因为 target_finetune_num 很小 (50)，完全可以常驻显存，避免循环内反复传输
    iter_target = iter(target_finetune_loader)
    num_iter_target = len(target_finetune_loader)
    finetune_trace_all = torch.zeros((num_iter_target, batch_size, 1, trace_length))

    # 将数据读入 Tensor
    for i in range(num_iter_target):
        finetune_trace_all[i, :, :, :], _ = next(iter_target)

    # 如果可用，直接放入 GPU
    if cuda:
        finetune_trace_all = finetune_trace_all.cuda()

    # get the number of batches
    num_iter = len(source_train_loader)
    clf_criterion = nn.CrossEntropyLoss()

    # 【优化】预先在 GPU 上生成 Discriminator 的标签（因为 Batch Size 固定）
    if cuda:
        real_label_static = torch.ones(batch_size, dtype=torch.long).cuda()
        fake_label_static = torch.zeros(batch_size, dtype=torch.long).cuda()
        critic_label_concat_static = torch.cat((real_label_static, fake_label_static), 0)
    else:
        real_label_static = torch.ones(batch_size, dtype=torch.long)
        fake_label_static = torch.zeros(batch_size, dtype=torch.long)
        critic_label_concat_static = torch.cat((real_label_static, fake_label_static), 0)

    # train on each batch of data
    for i in range(1, num_iter + 1):
        # get traces and labels for source domain
        source_data, source_label = next(iter_source)

        # 【优化】GPU 异步传输
        if cuda:
            source_data = source_data.cuda(non_blocking=True)
            source_label = source_label.cuda(non_blocking=True)

        # get traces for target domain
        # 直接从 GPU 上的 Tensor 索引，无需再次 .cuda()
        target_data = finetune_trace_all[(i - 1) % num_iter_target, :, :, :]

        # 移除 Variable

        ############################
        ### Train  Discriminator ###
        ############################
        # 【优化】set_to_none=True 更快
        optimizer_critic.zero_grad(set_to_none=True)

        # extract and concat features
        feat_s, output_s = atn_model(source_data)

        feat_t, output_t = atn_model(target_data)
        softmax_output_s = nn.Softmax(dim=1)(output_s)
        softmax_output_t = nn.Softmax(dim=1)(output_t)
        op_out_s = torch.bmm(softmax_output_s.unsqueeze(2), feat_s.unsqueeze(1))
        op_out_s = op_out_s.view(-1, softmax_output_s.size(1) * feat_s.size(1))
        op_out_t = torch.bmm(softmax_output_t.unsqueeze(2), feat_t.unsqueeze(1))
        op_out_t = op_out_t.view(-1, softmax_output_t.size(1) * feat_s.size(1))
        feat_concat = torch.cat((op_out_s, op_out_t), 0)
        # predict on discriminator
        pred_concat = critic(feat_concat.detach())

        # compute loss for critic using pre-allocated labels
        # 注意：如果最后一个 batch 不满，需要动态切片，但 drop_last=True 保证了 batch_size 固定
        loss_critic = clf_criterion(pred_concat, critic_label_concat_static)

        loss_critic.backward()
        # optimize critic
        optimizer_critic.step()

        preds = pred_concat.data.max(1, keepdim=True)[1]
        correct_batch = preds.eq(critic_label_concat_static.data.view_as(preds)).sum()

        ############################
        ### Train   the  Encoder ###
        ############################
        # zero gradients for optimizer
        optimizer_model.zero_grad(set_to_none=True)

        # extract target features
        feat_t, output_t = atn_model(target_data)
        softmax_output_t = nn.Softmax(dim=1)(output_t)
        op_out_t = torch.bmm(softmax_output_t.unsqueeze(2), feat_t.unsqueeze(1))
        op_out_t = op_out_t.view(-1, softmax_output_t.size(1) * feat_t.size(1))

        # predict on discriminator
        pred_t = critic(op_out_t)

        # compute adversarial discriminator loss (Trick Discriminator to think target is source)
        loss_tgt = clf_criterion(pred_t, real_label_static)

        # compute classification loss on source data
        _, pred_s = atn_model(source_data)
        if adjust_flag:
            pred_s = pred_s + 1 * adjustments

        loss_cls_s = clf_criterion(pred_s, source_label)

        total_loss = _lambda * loss_tgt + loss_cls_s
        total_loss.backward()

        # optimize total loss
        optimizer_model.step()

        if i % log_interval == 0:
            print(
                'Epoch {}: [{}/{} ({:.0f}%)]\tcritic_loss: {:.2f}\tencoder_loss: {:.2f}\tsource_cls_loss: {:.2f}\tcritic_acc: {:.2f}'.format(
                    epoch, i * len(source_data), len(source_train_loader) * batch_size,
                           100. * i / len(source_train_loader), loss_critic.item(),
                    loss_tgt.item(), loss_cls_s.item(), float(correct_batch) * 100. / (batch_size * 2)))


### Validation for ALPA
def ALPA_validation(atn_model, critic):
    """
    - atn_model: the adversarial transfer network
    - critic: the Discriminator
    """
    # enter evaluation mode
    atn_model.eval()
    critic.eval()

    # Instantiate the Iterator for source traces
    iter_source = iter(source_valid_loader)

    # 【优化】同样预加载 Target Validation 数据到 GPU
    iter_target = iter(target_finetune_loader)
    num_iter_target = len(target_finetune_loader)
    finetune_trace_all = torch.zeros((num_iter_target, batch_size, 1, trace_length))
    for i in range(num_iter_target):
        finetune_trace_all[i, :, :, :], _ = next(iter_target)
    if cuda:
        finetune_trace_all = finetune_trace_all.cuda()

    num_iter = len(source_valid_loader)

    # 预分配 label
    if cuda:
        real_label_static = torch.ones(batch_size, dtype=torch.long).cuda()
    else:
        real_label_static = torch.ones(batch_size, dtype=torch.long)

    total_tgt_loss = 0
    total_cls_loss = 0
    clf_criterion = nn.CrossEntropyLoss()

    # 【优化】Validation 不需要梯度
    with torch.no_grad():
        for i in range(1, num_iter + 1):
            # get traces and labels for source domain
            source_data, source_label = next(iter_source)
            # get traces for target domain (from GPU memory)
            target_data = finetune_trace_all[(i - 1) % num_iter_target, :, :, :]

            if cuda:
                source_data = source_data.cuda(non_blocking=True)
                source_label = source_label.cuda(non_blocking=True)

            ############################
            # extract and target features
            feat_t, output_t = atn_model(target_data)
            softmax_output_t = nn.Softmax(dim=1)(output_t)

            op_out_t = torch.bmm(softmax_output_t.unsqueeze(2), feat_t.unsqueeze(1))
            op_out_t = op_out_t.view(-1, softmax_output_t.size(1) * feat_t.size(1))

            # predict on discriminator
            pred_t = critic(op_out_t)

            # compute adversarial discriminator loss
            total_tgt_loss += clf_criterion(pred_t, real_label_static).item()

            # compute classification loss on source data
            _, pred_s = atn_model(source_data)
            total_cls_loss += clf_criterion(pred_s, source_label).item()

    total_tgt_loss /= len(source_valid_loader)
    total_cls_loss /= len(source_valid_loader)
    total_loss = _lambda * total_tgt_loss + total_cls_loss

    print('Validation: total_loss: {:.4f}, encoder_loss: {:.4f}, clf_loss:{:.4f}'.format(
        total_loss, total_tgt_loss, total_cls_loss))
    return total_loss, total_tgt_loss, total_cls_loss


### the Adversarial transfer network
class ATN(nn.Module):
    def __init__(self, num_classes=9):
        super(ATN, self).__init__()
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
            nn.SELU(),
        )
        # the output layer
        self.final_classifier = nn.Sequential(
            nn.Linear(20, num_classes)
        )

    # how the network runs
    def forward(self, input):
        x = self.features(input)
        feature = x.view(x.size(0), -1)
        # print(feature.shape)
        output = self.classifier_1(feature)
        output = self.final_classifier(output)
        return feature, output


### the discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # the discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(64 * 9, 64),
            nn.SELU(),
            nn.Linear(64, 2)
        )

    # how the network runs
    def forward(self, input):
        output = self.discriminator(input)
        return output

def compute_adjustment_1(Y_profiling, tro, classes=9, eps=1e-12):
    counts = np.bincount(Y_profiling, minlength=classes).astype(np.float64)
    pi = counts / np.maximum(counts.sum(), 1.0)
    adjustments = np.log(np.power(pi, tro) + eps)  # log(pi^tau)
    return adjustments.astype(np.float32)

if __name__ == '__main__':
    DEVICE_CONFIG = {
        i: {'key': i, 'folder': f'device{i:02d}'}
        for i in range(1, 9)
    }

    source_device_id = 1  # 源域设备 ID
    target_device_id = 3  # 目标域设备 ID

    if source_device_id not in DEVICE_CONFIG or target_device_id not in DEVICE_CONFIG:
        raise ValueError("设备ID必须在 1-8 之间")

    # 自动获取
    real_key_01 = DEVICE_CONFIG[source_device_id]['key']
    real_key_02 = DEVICE_CONFIG[target_device_id]['key']

    source_file_path = f"./Data/{DEVICE_CONFIG[source_device_id]['folder']}/"
    target_file_path = f"./Data/{DEVICE_CONFIG[target_device_id]['folder']}/"

    print(f"Source: Device {source_device_id} | Path: {source_file_path} | Key: {hex(real_key_01)}")
    print(f"Target: Device {target_device_id} | Path: {target_file_path} | Key: {hex(real_key_02)}")

    labeling_method = 'hw'  # labeling of trace
    _lambda = 0.1
    batch_size = 100
    total_epoch = 50
    finetune_epoch = 50  # epoch number for fine-tuning
    lr = 0.001  # learning rate
    log_interval = 40  # epoch interval to log training information
    train_num = 20000
    valid_num = 5000
    source_test_num = 5000
    target_finetune_num = batch_size
    target_test_num = 4500
    trace_offset = 0
    trace_length = 500

    no_cuda = False
    cuda = not no_cuda and torch.cuda.is_available()
    seed = 8
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    if labeling_method == 'identity':
        class_num = 256
    elif labeling_method == 'hw':
        class_num = 9

    # to load traces and labels
    X_train_source = np.load(source_file_path + 'X_train.npy')
    Y_train_source = np.load(source_file_path + 'Y_train.npy')
    X_attack_target = np.load(target_file_path + 'X_attack.npy')
    Y_attack_target = np.load(target_file_path + 'Y_attack.npy')

    # to load plaintexts
    plaintexts_source = np.load(source_file_path + 'plaintexts_attack.npy')
    plaintexts_target = np.load(target_file_path + 'plaintexts_attack.npy')
    plaintexts_target = plaintexts_target[target_finetune_num:target_finetune_num + target_test_num]

    adjustments_np = compute_adjustment_1(Y_train_source[0:train_num], tro=1, classes=9)
    adjustments = torch.from_numpy(adjustments_np).view(1, -1)
    if cuda:
        adjustments = adjustments.cuda()

    # horizontal_standardization
    mn = np.repeat(np.mean(X_train_source, axis=1, keepdims=True), X_train_source.shape[1], axis=1)
    std = np.repeat(np.std(X_train_source, axis=1, keepdims=True), X_train_source.shape[1], axis=1)
    X_train_source = (X_train_source - mn) / std

    mn = np.repeat(np.mean(X_attack_target, axis=1, keepdims=True), X_attack_target.shape[1], axis=1)
    std = np.repeat(np.std(X_attack_target, axis=1, keepdims=True), X_attack_target.shape[1], axis=1)
    X_attack_target = (X_attack_target - mn) / std

    # parameters of data loader
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
    kwargs_target_finetune = {
        'trs_file': X_attack_target[0:target_finetune_num, :],
        'label_file': Y_attack_target[0:target_finetune_num],
        'trace_num': target_finetune_num,
        'trace_offset': trace_offset,
        'trace_length': trace_length,
    }

    source_train_loader = load_training(batch_size, kwargs_source_train)
    source_valid_loader = load_training(batch_size, kwargs_source_valid)
    target_finetune_loader = load_training(batch_size, kwargs_target_finetune)
    print('Load data complete!')

    # create a network
    model = ATN(num_classes=9)
    discriminator = Discriminator()
    print('Construct model complete')
    if cuda:
        model.cuda()
        discriminator.cuda()
    # initialize a big enough loss
    min_loss = 1000
    # load the pre-trained network
    # 确保文件存在再加载，否则可能会报错
    adjust_flag = True

    if adjust_flag:
        flag = "real"
    else:
        flag = "fake"
    pretrained_path = './models/'+str(flag)+'_pre-trained_cpda_device{}.pth'.format(source_device_id)
    print(pretrained_path)
    if os.path.exists(pretrained_path):
        checkpoint = torch.load(pretrained_path)
        model_dict = checkpoint['model_state_dict']
        model.load_state_dict(model_dict)
    else:
        print(f"Warning: Pretrained model not found at {pretrained_path}")

    optimizer_critic = optim.SGD([
        {'params': discriminator.discriminator.parameters()},
    ], lr=lr, weight_decay=0.0005, momentum=0.9)
    optimizer_model = optim.SGD([
        {'params': model.features.parameters()},
        {'params': model.classifier_1.parameters()},
        {'params': model.final_classifier.parameters()},
    ], lr=lr, weight_decay=0.0005, momentum=0.9)

    start_time = time.time()
    end_time = time.time()

    # restore the optimizer state
    for epoch in range(1, finetune_epoch + 1):
        print(f'Train Epoch {epoch}:')
        ALPA_train(epoch, model, discriminator)

        # Validation
        val_total_loss, val_tgt_loss, val_cls_loss = ALPA_validation(model, discriminator)

        if (val_total_loss < min_loss):
            min_loss = val_total_loss
            # 确保目录存在
            if not os.path.exists('models'):
                os.makedirs('models')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
            }, './models/alpa'+str(adjust_flag)+str(target_finetune_num)+'final_device{}_to_{}.pth'.format(source_device_id, target_device_id))
            end_time = time.time()



    elapsed_time = end_time - start_time
    print(f"\n总训练时间: {elapsed_time:.2f} 秒")
    # 如果需要更友好的格式，可以转换为分钟和秒
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60
    print(f"总训练时间: {minutes} 分 {seconds:.2f} 秒")