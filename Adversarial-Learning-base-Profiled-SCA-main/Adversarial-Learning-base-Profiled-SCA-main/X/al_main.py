import os
from torch.autograd import Variable
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from torch.utils.data import Dataset, DataLoader
import torch
from torch import optim
# from torch.autograd import Variable # Variable 已废弃，移除以加速
import numpy as np
import math
from torch import nn
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import matplotlib.pyplot as plt
import itertools
import random
from torchvision import transforms

### handle the dataset
class TorchDataset(Dataset):
    def __init__(self, trs_file, label_file, trace_num, trace_offset, trace_length):
        self.trs_file = trs_file
        self.label_file = label_file
        self.trace_num = trace_num
        self.trace_offset = trace_offset
        self.trace_length = trace_length
        self.ToTensor = transforms.ToTensor()
    def __getitem__(self, i):
        index = i % self.trace_num
        trace = self.trs_file[index,:]
        label = self.label_file[index]
        trace = trace[self.trace_offset:self.trace_offset+self.trace_length]
        trace = np.reshape(trace,(1,-1))
        trace = self.ToTensor(trace)
        trace = np.reshape(trace, (1,-1))
        label = torch.tensor(label, dtype=torch.long)
        return trace.float(), label
    def __len__(self):
        return self.trace_num

def load_training(batch_size, kwargs):
    data = TorchDataset(**kwargs)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1, pin_memory=True)
    return train_loader

### data loader for testing
def load_testing(batch_size, kwargs):
    data = TorchDataset(**kwargs)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=1, pin_memory=True)
    return test_loader

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# --- 加速点 1: 开启 cuDNN 自动调优 ---
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


### 简化后的 Dataset (因为数据已经在 GPU 上了，这里只需要简单的索引)
# 实际上，优化后我们可以直接使用 PyTorch 自带的 TensorDataset，
# 但为了保持代码结构，我们稍微修改一下你的类，让它只负责返回数据
class FastTensorDataset(Dataset):
    def __init__(self, data_tensor, label_tensor):
        self.data_tensor = data_tensor
        self.label_tensor = label_tensor
        self.trace_num = len(data_tensor)

    def __getitem__(self, index):
        return self.data_tensor[index], self.label_tensor[index]

    def __len__(self):
        return self.trace_num


### data loader
# --- 加速点 2: num_workers=0 ---
# 因为数据已经全在 GPU 上，不需要多进程加载，单进程最快且无兼容性问题
def get_loader(data_tensor, label_tensor, batch_size, shuffle=True):
    dataset = FastTensorDataset(data_tensor, label_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True, num_workers=0)
    return loader


# Sbox 和 HW_byte 表保持不变
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


### ALPA
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
    # Instantiate the Iterator for target traces
    iter_target = iter(target_finetune_loader)
    num_iter_target = len(target_finetune_loader)
    finetune_trace_all = torch.zeros((num_iter_target, batch_size, 1, trace_length))
    for i in range(num_iter_target):
        finetune_trace_all[i, :, :, :], _ = next(iter_target)
    # get the number of batches
    num_iter = len(source_train_loader)
    clf_criterion = nn.CrossEntropyLoss()
    # train on each batch of data
    for i in range(1, num_iter + 1):
        # get traces and labels for source domain
        source_data, source_label = next(iter_source)
        # get traces for target domain
        target_data = finetune_trace_all[(i - 1) % num_iter_target, :, :, :]
        # Instantiate the target Iterator again if all target traces have been used
        if cuda:
            source_data = source_data.cuda()
            source_label = source_label.cuda()
            target_data = target_data.cuda()
        source_data = Variable(source_data)
        target_data = Variable(target_data)
        ############################
        ### Train  Discriminator ###
        ############################
        optimizer_critic.zero_grad()
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

        # prepare domain labels, 1 for source device, 0 for target device
        critic_label_s = Variable((torch.ones(feat_s.size(0)).long()).cuda())
        critic_label_t = Variable((torch.zeros(feat_t.size(0)).long()).cuda())
        critic_label_concat = torch.cat((critic_label_s, critic_label_t), 0)
        # compute loss for critic
        loss_critic = clf_criterion(pred_concat, critic_label_concat)
        loss_critic.backward()
        # optimize critic
        optimizer_critic.step()
        preds = pred_concat.data.max(1, keepdim=True)[1]
        # get the number of correct prediction
        correct_batch = preds.eq(critic_label_concat.data.view_as(preds)).sum()

        ############################
        ### Train   the  Encoder ###
        ############################
        # zero gradients for optimizer
        optimizer_model.zero_grad()
        # extract target features
        feat_t, output_t = atn_model(target_data)
        softmax_output_t = nn.Softmax(dim=1)(output_t)
        op_out_t = torch.bmm(softmax_output_t.unsqueeze(2), feat_t.unsqueeze(1))
        op_out_t = op_out_t.view(-1, softmax_output_t.size(1) * feat_t.size(1))
        # predict on discriminator
        pred_t = critic(op_out_t)
        # prepare fake labels
        fake_label_t = Variable((torch.ones(feat_t.size(0)).long()).cuda())
        # compute adversarial discriminator loss
        loss_tgt = clf_criterion(pred_t, fake_label_t)
        # compute classification loss on source data
        _, pred_s = atn_model(source_data)
        loss_cls_s = clf_criterion(pred_s, source_label)
        total_loss = _lambda * loss_tgt + loss_cls_s
        total_loss.backward()
        # optimize total loss
        optimizer_model.step()
        if i % log_interval == 0:
            print(
                'Epoch {}: [{}/{} ({:.0f}%)]\tcritic_loss: {:.2f}\tencoder_loss: {:.2f}\tsource_cls_loss: {:.2f}\tcritic_acc: {:.2f}'.format(
                    epoch, i * len(source_data), len(source_train_loader) * batch_size,
                           100. * i / len(source_train_loader), loss_critic.data,
                    loss_tgt.data, loss_cls_s.data, float(correct_batch) * 100. / (batch_size * 2)))


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
    # Instantiate the Iterator for target traces
    iter_target = iter(target_finetune_loader)

    num_iter_target = len(target_finetune_loader)
    finetune_trace_all = torch.zeros((num_iter_target, batch_size, 1, trace_length))
    clf_criterion = nn.CrossEntropyLoss()
    for i in range(num_iter_target):
        finetune_trace_all[i, :, :, :], _ = iter_target.next()
    # get the number of batches
    num_iter = len(source_valid_loader)
    # the adversarial discriminator loss
    total_tgt_loss = 0
    # the classification loss
    total_cls_loss = 0
    for i in range(1, num_iter + 1):
        # get traces and labels for source domain
        source_data, source_label = iter_source.next()
        # get traces for target domain
        target_data = finetune_trace_all[(i - 1) % num_iter_target, :, :, :]
        # Instantiate the target Iterator again if all target traces have been used
        if cuda:
            source_data = source_data.cuda()
            source_label = source_label.cuda()
            target_data = target_data.cuda()
        source_data = Variable(source_data)
        target_data = Variable(target_data)
        ############################
        # extract and target features
        feat_t, output_t = atn_model(target_data)
        softmax_output_t = nn.Softmax(dim=1)(output_t)
        op_out_t = torch.bmm(softmax_output_t.unsqueeze(2), feat_t.unsqueeze(1))
        op_out_t = op_out_t.view(-1, softmax_output_t.size(1) * feat_t.size(1))
        # predict on discriminator
        pred_t = critic(op_out_t)
        # prepare fake labels
        fake_label_t = Variable((torch.ones(feat_t.size(0)).long()).cuda())
        # compute adversarial discriminator loss
        total_tgt_loss = total_tgt_loss + nn.CrossEntropyLoss()(pred_t, fake_label_t)
        # compute classification loss on source data
        _, pred_s = atn_model(source_data)
        total_cls_loss = total_cls_loss + clf_criterion(pred_s, source_label)
    total_tgt_loss /= len(source_valid_loader)
    total_cls_loss /= len(source_valid_loader)
    total_loss = _lambda * total_tgt_loss + total_cls_loss
    # total_loss = total_tgt_loss + total_cls_loss
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


if __name__ == '__main__':  # <--- 添加这一行
    source_device_id = 1
    target_device_id = 2
    real_key_01 = 0x01  # key of the source domain
    real_key_02 = 0x02  # key of the target domain
    labeling_method = 'hw'  # labeling of trace
    _lambda = 0.05
    batch_size = 50
    total_epoch = 100
    finetune_epoch = 200  # epoch number for fine-tuning
    lr = 0.001  # learning rate
    log_interval = 40  # epoch interval to log training information
    train_num = 20000
    valid_num = 5000
    source_test_num = 5000
    target_finetune_num = 50
    target_test_num = 4500
    trace_offset = 0
    trace_length = 500
    source_file_path = 'Data/device01/'
    target_file_path = 'Data/device02/'
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
    X_attack_source = np.load(source_file_path + 'X_attack.npy')
    Y_attack_source = np.load(source_file_path + 'Y_attack.npy')
    X_attack_target = np.load(target_file_path + 'X_attack.npy')
    Y_attack_target = np.load(target_file_path + 'Y_attack.npy')

    # horizontal_standardization
    mn = np.repeat(np.mean(X_train_source, axis=1, keepdims=True), X_train_source.shape[1], axis=1)
    std = np.repeat(np.std(X_train_source, axis=1, keepdims=True), X_train_source.shape[1], axis=1)
    X_train_source = (X_train_source - mn) / std

    mn = np.repeat(np.mean(X_attack_source, axis=1, keepdims=True), X_attack_source.shape[1], axis=1)
    std = np.repeat(np.std(X_attack_source, axis=1, keepdims=True), X_attack_source.shape[1], axis=1)
    X_attack_source = (X_attack_source - mn) / std

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
    kwargs_source_test = {
        'trs_file': X_attack_source,
        'label_file': Y_attack_source,
        'trace_num': source_test_num,
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
    kwargs_target = {
        'trs_file': X_attack_target[target_finetune_num:target_finetune_num + target_test_num, :],
        'label_file': Y_attack_target[target_finetune_num:target_finetune_num + target_test_num],
        'trace_num': target_test_num,
        'trace_offset': trace_offset,
        'trace_length': trace_length,
    }
    source_train_loader = load_training(batch_size, kwargs_source_train)
    source_valid_loader = load_training(batch_size, kwargs_source_valid)
    source_test_loader = load_testing(batch_size, kwargs_source_test)
    target_finetune_loader = load_training(batch_size, kwargs_target_finetune)
    target_test_loader = load_testing(batch_size, kwargs_target)
    print('Load data complete!')

    model = ATN(num_classes=9)
    discriminator = Discriminator()
    print('Construct model complete')
    if cuda:
        model.cuda()
        discriminator.cuda()
    # initialize a big enough loss
    min_loss = 1000
    # load the pre-trained network
    checkpoint = torch.load('./models/pre-trained_device{}.pth'.format(source_device_id))
    model_dict = checkpoint['model_state_dict']
    model.load_state_dict(model_dict)
    optimizer_critic = optim.SGD([
        {'params': discriminator.discriminator.parameters()},
    ], lr=lr, weight_decay=0.0005, momentum=0.9)
    optimizer_model = optim.SGD([
        {'params': model.features.parameters()},
        {'params': model.classifier_1.parameters()},
        {'params': model.final_classifier.parameters()},
    ], lr=lr, weight_decay=0.0005, momentum=0.9)
    # restore the optimizer state
    for epoch in range(1, finetune_epoch + 1):
        print(f'Train Epoch {epoch}:')
        ALPA_train(epoch, model, discriminator)
        val_total_loss, val_tgt_loss, val_cls_loss = ALPA_validation(model, discriminator)
        if (val_total_loss < min_loss):
            min_loss = val_total_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
            }, './models/final_device{}_to_{}.pth'.format(source_device_id, target_device_id))