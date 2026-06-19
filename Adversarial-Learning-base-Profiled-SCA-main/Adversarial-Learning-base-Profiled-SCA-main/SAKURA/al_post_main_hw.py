from torch.utils.data import Dataset
import torch
from torch import optim
import numpy as np
from torch import nn
torch.backends.cudnn.benchmark = True
import os
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

HW_byte = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2,
           3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2, 3,
           3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3,
           4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4,
           3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5,
           6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4,
           4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5,
           6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8]

# 【优化】将 HW_byte 转换为 Numpy 数组，支持向量化索引
HW_byte_np = np.array(HW_byte, dtype=np.int32)


def ALPA_train(epoch, atn_model, critic):
    """
    - epoch : the current epoch
    - atn_model: the adversarial transfer network
    - critic: the Discriminator
    """
    # enter training mode
    atn_model.eval()
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

        total_loss = lambda_ * loss_tgt + loss_cls_s
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
    total_loss = lambda_ * total_tgt_loss + total_cls_loss

    print('Validation: total_loss: {:.4f}, encoder_loss: {:.4f}, clf_loss:{:.4f}'.format(
        total_loss, total_tgt_loss, total_cls_loss))
    return total_loss, total_tgt_loss, total_cls_loss

### the fine-tuning model
class CDP_Net(nn.Module):
    def __init__(self, num_classes=9):
        super(CDP_Net, self).__init__()
        # the encoder part
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
        # the fully-connected layer 1
        self.classifier_1 = nn.Sequential(
            nn.Linear(448, 2),
            nn.SELU(),
        )
        # the output layer
        self.final_classifier = nn.Sequential(
            nn.Linear(2, num_classes)
        )
    # how the network runs
    def forward(self, input):
        x = self.features(input)
        feature = x.view(x.size(0), -1)
        output = self.classifier_1(feature)
        output = self.final_classifier(output)
        return feature, output

### the discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # the discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(448 * 9, 64),
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


def calculate_HW(data):
    # 【优化】使用向量化索引代替列表推导式，速度极快
    return HW_byte_np[data.astype(int)]

if __name__ == '__main__':
    source_device_id = 1
    target_device_id = 2
    # roundkeys of the three devices are : 0x21, 0xCD, 0x8F
    real_key_01 = 0x21  # key of the source domain
    real_key_02 = 0xCD  # key of the target domain
    labeling_method = 'hw'
    batch_size = 200
    total_epoch = 100
    finetune_epoch = 30
    lr = 0.001
    log_interval = 50
    train_num = 85000
    valid_num = 5000
    source_test_num = 9900
    target_finetune_num = batch_size
    target_test_num = 9400
    trace_offset = 0
    lambda_ = 0.08  # Penalty coefficient
    trace_length = 1000
    source_file_path = './Data/device1/'
    target_file_path = './Data/device2/'
    no_cuda = False
    cuda = not no_cuda and torch.cuda.is_available()
    seed = 8
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    X_train_source = np.load(source_file_path + 'X_train.npy')
    Y_train_source = np.load(source_file_path + 'Y_train.npy')
    X_attack_target = np.load(target_file_path + 'X_attack.npy')
    Y_attack_target = np.load(target_file_path + 'Y_attack.npy')

    if labeling_method == 'identity':
        class_num = 256
    elif labeling_method == 'hw':
        class_num = 9
        # 【优化】使用向量化函数
        Y_train_source = calculate_HW(Y_train_source)
        Y_attack_target = calculate_HW(Y_attack_target)

    mn = np.repeat(np.mean(X_train_source, axis=1, keepdims=True), X_train_source.shape[1], axis=1)
    std = np.repeat(np.std(X_train_source, axis=1, keepdims=True), X_train_source.shape[1], axis=1)
    X_train_source = (X_train_source - mn) / std

    mn = np.repeat(np.mean(X_attack_target, axis=1, keepdims=True), X_attack_target.shape[1], axis=1)
    std = np.repeat(np.std(X_attack_target, axis=1, keepdims=True), X_attack_target.shape[1], axis=1)
    X_attack_target = (X_attack_target - mn) / std

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

    adjustments_np = compute_adjustment_1(Y_train_source[0:train_num], tro=1, classes=9)

    adjustments = torch.from_numpy(adjustments_np).view(1, -1)
    if cuda:
        adjustments = adjustments.cuda()

    adjust_flag = True

    if adjust_flag:
        flag = "real"
    else:
        flag = "fake"
    pretrained_path = ('./models/hw'+str(flag)+'_pre-trained_cpda_device{}.pth'.format(source_device_id))

    device = torch.device('cuda' if cuda else 'cpu')

    model = CDP_Net(num_classes=9).to(device)
    discriminator = Discriminator().to(device)  # ✅ 关键

    # 如果你用 adjustments
    adjustments = adjustments.to(device)
    print("Loading model:", pretrained_path)
    # 【优化】增加文件存在性检查，避免 Crash
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
        {'params': model.final_classifier.parameters()}
    ], lr=lr)

    if cuda:
        model.cuda()
    min_loss = 1000

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
            }, './models/alpa_hw_'+str(flag)+'final_device{}_to_{}.pth'.format(source_device_id, target_device_id))
    del source_train_loader
    del source_valid_loader
    del target_finetune_loader

    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    import gc

    gc.collect()

    print("Cleanup complete, exiting...")