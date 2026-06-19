from data_load_ascad_50000 import read_data
from SCA_util import perform_attacks
import DL_model
import numpy as np
import tensorflow as tf
import tensorflow.keras as tk
from datetime import datetime
from clr import OneCycleLR
import numpy as np

import matplotlib.pyplot as plt


class all(tf.keras.callbacks.Callback):
    def __init__(self, validation=None, grad_data=None, classes=9):
        super(all, self).__init__()
        self.validation = validation
        self.grad_data = grad_data  # 用于计算梯度的数据
        self.class_grads_history = []  # 存储梯度历史
        self.classes = classes  # 类别数量

    def on_epoch_end(self, epoch, logs=None):
        # 原有的验证逻辑...
        if self.validation:
            global best_weights
            global count
            global epoch_count
            logs['all_val'] = float('inf')
            X_attack_valid_metric, all_valid_plt_attack_metric = self.validation[0], self.validation[1]
            y_pred_valid_metric = self.model.predict(X_attack_valid_metric)
            # y_pred_valid_metric = y_pred_valid_metric -0.25 * adjustments
            y_pred_valid_metric = tf.nn.softmax(y_pred_valid_metric)
            epoch_count = epoch_count + 1
            avg_rank_current, avg_attack_traces, avg_corr_current = perform_attacks(all_valid_plt_attack_metric,
                                                                                    y_pred_valid_metric,
                                                                                    'all',
                                                                                    leakage_model, dataset,
                                                                                    num_traces_attacks)
            all_corr_logs.append(avg_corr_current)
            if avg_attack_traces[-1, correct_key] > 0:
                print("攻击失败", )
                print("GE:", avg_attack_traces[-1, correct_key])
                print("corr:", avg_corr_current)

            else:
                print("攻击成功")
                print("TGE0:", np.argmax(avg_attack_traces[:, correct_key] < 1))
                print("corr:", avg_corr_current)

            if not corr_logs:
                corr_logs.append(avg_corr_current)
                best_weights = self.model.get_weights()
            else:
                if corr_logs[-1] < avg_corr_current:
                    corr_logs.append(avg_corr_current)
                    best_weights = self.model.get_weights()
                    count = 0
                else:
                    count = count + 1
                    print(count)
                    if count == 10:
                        self.model.stop_training = True
                        self.model.set_weights(best_weights)

        # 新增梯度计算逻辑
        # 在 all 类的 on_epoch_end 方法中:
        # --- MODIFIED GRADIENT CALCULATION LOGIC FOR PARAMETER GRADIENT PERCENTAGE ---
        if self.grad_data:
            X_grad_input, Y_grad_input = self.grad_data
            X_grad_tf = tf.convert_to_tensor(X_grad_input, dtype=tf.float32)
            # Y_true_for_loss_full_batch is for the entire grad_data batch
            Y_true_for_loss_full_batch = tf.convert_to_tensor(Y_grad_input[:, :self.classes], dtype=tf.float32)
            y_true_labels_int = np.argmax(Y_true_for_loss_full_batch.numpy(), axis=1)

            model_trainable_vars = self.model.trainable_variables
            if not model_trainable_vars:  # No trainable variables
                print(f"Epoch {epoch + 1}: No trainable variables in the model.")
                # Populate with zeros and continue
                current_epoch_param_grad_percentages = {}
                for cls_idx in range(self.classes):
                    current_epoch_param_grad_percentages[f'class_{cls_idx}_param_grad_norm_percentage'] = 0.0
                self.class_grads_history.append(current_epoch_param_grad_percentages)
                return  # Exit gradient calculation if no trainable vars

            # --- Step 1: Calculate norm of parameter gradients for each true class's samples ---
            norm_of_param_grads_per_true_class = np.zeros(self.classes)

            for cls_idx in range(self.classes):  # cls_idx here represents the true class C
                mask = (y_true_labels_int == cls_idx)

                if np.sum(mask) > 0:
                    X_class_subset = X_grad_tf[mask]
                    Y_class_subset_one_hot = tf.gather(Y_true_for_loss_full_batch,
                                                       tf.where(mask)[:, 0])  # Get corresponding Y for subset

                    with tf.GradientTape() as tape_class_specific:
                        # tape_class_specific.watch(model_trainable_vars) # Ensure variables are watched
                        logits_class_subset = self.model(X_class_subset, training=False)
                        loss_class_subset = adjustment_loss(Y_class_subset_one_hot, logits_class_subset)

                    # Gradients of this class's loss w.r.t. model parameters
                    param_grads_for_class = tape_class_specific.gradient(loss_class_subset, model_trainable_vars)

                    if param_grads_for_class and any(g is not None for g in param_grads_for_class):
                        # Filter out None gradients (e.g., for non-trainable layers if any were accidentally included)
                        valid_param_grads_for_class = [g for g in param_grads_for_class if g is not None]
                        if valid_param_grads_for_class:
                            # Calculate the L2 norm of the list of gradient tensors
                            norm_of_param_grads_per_true_class[cls_idx] = tf.linalg.global_norm(
                                valid_param_grads_for_class).numpy()
                # else it remains 0.0 for classes not present or if grads are None

            # --- Step 2: Calculate total sum of these norms ---
            total_sum_of_param_grad_norms = np.sum(norm_of_param_grads_per_true_class)

            # --- Step 3: Calculate percentage contribution ---
            current_epoch_param_grad_percentages = {}
            for cls_idx in range(self.classes):
                percentage = 0.0
                if total_sum_of_param_grad_norms > 1e-9:  # Avoid division by zero
                    percentage = (norm_of_param_grads_per_true_class[cls_idx] / total_sum_of_param_grad_norms) * 100.0

                current_epoch_param_grad_percentages[f'class_{cls_idx}_param_grad_norm_percentage'] = percentage

            self.class_grads_history.append(current_epoch_param_grad_percentages)
        # --- END OF MODIFIED GRADIENT CALCULATION LOGIC ---
        # --- END OF MODIFIED GRADIENT CALCULATION LOGIC ---


def compute_adjustment(Y_profiling, tro):
    """compute the base probabilities"""

    Y_profiling = np.argmax(Y_profiling[:, :9], 1)
    label_freq = {}
    for key in Y_profiling:
        label_freq[key] = label_freq.get(key, 0) + 1
    label_freq = dict(sorted(label_freq.items()))
    label_freq_array = np.array(list(label_freq.values()))
    label_freq_array = label_freq_array / label_freq_array.sum()
    adjustments = np.log(label_freq_array ** tro + 1e-12)

    # adaptive_margin =  1.0 / np.sqrt(np.sqrt(label_freq_array))
    # adaptive_margin = adaptive_margin * (0.5 / np.max(adaptive_margin))
    # adaptive_margin = tf.cast(adaptive_margin, dtype=tf.float32)
    return adjustments


def adjustment_loss(y_true, y_pred):
    y_true = y_true[:, :classes]

    if adjust_flag:
        # 确保adjustments与y_pred类型一致
        adjusted = tf.cast(adjustments, dtype=y_pred.dtype)  # 新增类型转换
        y_pred = y_pred + 1 * adjusted  # 使用转换后的调整量

    y_pred = tf.nn.softmax(y_pred, axis=1)
    loss = tk.backend.categorical_crossentropy(y_true, y_pred)
    return loss


if __name__ == '__main__':

    """变量配置"""
    dataset = 'ASCAD'  # ASCAD/ASCAD_rand/CHES_CTF
    leakage_model = 'HW'
    attack_model = 'CNN'  # MLP/CNN
    sigma_hw = 0  # sigma for the HW leakage model
    sigma_id = 0  # sigma for the ID leakage model
    num_traces_attacks = 1500
    epoch_count = 0
    # Select leakage model
    if leakage_model == 'ID':
        classes = 256
    else:
        classes = 9
    data_arguementation = False  # enable/disbale data arguementation
    data_arguementation_level = 0.25  # data arguementation level
    epoch = 50
    learning_rate = 5e-3
    output_metric = "all"  # rank/corr
    companion_metric = None  # None/all/kl_loss_model/'categorical_accuracy'
    model_size = 64  # the size of the profiling model
    rank_logs = []
    all_rank_logs = []
    corr_logs = []
    all_corr_logs = []
    loss_logs = []
    kl_loss_logs = []
    count = 0
    best_weights = None
    tro = 1
    adjust_flag = True
    current_time = datetime.now()
    day = current_time.day
    hour = current_time.hour
    minute = current_time.minute
    experiment_time = 'time_is{}_{}_{}'.format(int(day),
                                               int(hour),
                                               int(minute)
                                               )

    """数据导入"""
    (X_profiling, X_attack), (Y_profiling, Y_attack), (
        plt_profiling, plt_attack), correct_key, attack_byte, num_profiling_traces = read_data(
        leakage_model,
        data_arguementation,
        data_arguementation_level,
        attack_model, dataset,
        sigma_hw, sigma_id)

    # 确保标签数据为float32
    Y_profiling = Y_profiling.astype(np.float32)  # 新增类型转换
    Y_attack = Y_attack.astype(np.float32)  # 新增类型转换

    adjustments = compute_adjustment(Y_profiling, tro)

    """创建神经网络模型"""
    """ select the output_metric function """

    #
    model, batch_size, epoch_sota = DL_model.pick_SOAT(dataset, leakage_model, X_profiling.shape[1],
                                                       companion_metric,
                                                       adjustment_loss, learning_rate, model=attack_model,
                                                       model_size=model_size)
    lr_manager = OneCycleLR(len(X_attack[:5000]), 128, learning_rate, end_percentage=0.2, scale_percentage=0.1,
                            maximum_momentum=None, minimum_momentum=None, verbose=True)

    grad_data = (X_profiling[:5000], Y_profiling[:5000])  # 使用部分训练数据计算梯度
    callback = [
        lr_manager,
        all(validation=(X_attack[:5000], plt_attack[:5000], Y_attack[:5000]),
            grad_data=grad_data,
            classes=classes)
    ]

    """最佳模型的存储地址"""
    test_info = 'dataset{}_leakage_model{}_epoch{}_batch_size{}_output{}'.format(dataset, leakage_model,
                                                                                 epoch,
                                                                                 batch_size,
                                                                                 output_metric,
                                                                                 )
    model_root = 'Model/'

    filename = model_root + test_info
    """开始训练"""
    history = model.fit(x=X_profiling[:50000], y=Y_profiling[:50000, :9], batch_size=128, verbose=2,
                        epochs=epoch,
                        callbacks=callback
                        )
    """梯度可视化"""
    # --- MODIFIED GRADIENT VISUALIZATION ---
    # --- MODIFIED GRADIENT VISUALIZATION FOR PERCENTAGE ---
    # ... (所有之前的 if __name__ == '__main__' 代码，直到梯度可视化之前，都保持不变) ...

    """梯度可视化"""
    # --- MODIFIED GRADIENT VISUALIZATION FOR PARAMETER GRADIENT NORM PERCENTAGE ---
    grad_callback_instance = None
    for cb_item in callback:  # callback_list 是您代码中定义的回调列表
        if isinstance(cb_item, all):  # all 是您自定义的回调类名
            grad_callback_instance = cb_item
            break

    if grad_callback_instance and grad_callback_instance.class_grads_history:
        plt.figure(figsize=(14, 9))

        # 'classes' 是在您的脚本前面定义的全局变量
        colors = plt.cm.get_cmap('tab20', classes)

        for cls_idx_plot in range(classes):
            # 检索新的百分比度量：参数梯度范数贡献占比
            param_grad_norm_percentage_values = [
                epoch_data.get(f'class_{cls_idx_plot}_param_grad_norm_percentage', 0.0)  # 使用新的键名
                for epoch_data in grad_callback_instance.class_grads_history
            ]

            # 使用之前定义的颜色逻辑
            current_color = colors(cls_idx_plot % colors.N)  # colors.N 是 colormap 中的颜色数量

            plt.plot(param_grad_norm_percentage_values,
                     label=f'Class {cls_idx_plot} Param Grad Norm Pct.',  # 更新图例标签
                     color=current_color,
                     linewidth=2)

        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Parameter Gradient Norm Contribution Pct. (%)', fontsize=11)  # 更新Y轴标签
        plt.title('Pct. Contrib. of True Class Samples to Parameter Gradient Norm', fontsize=13)  # 更新标题

        plt.legend(bbox_to_anchor=(1.03, 1), loc='upper left', fontsize='small')
        plt.grid(True, alpha=0.4)
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # 调整布局为图例腾出空间

        # 'experiment_time' 是在您的脚本前面定义的变量
        save_filename = f'param_grad_norm_percentage_analysis_{experiment_time}.png'  # 更新保存文件名
        plt.savefig(save_filename, dpi=300)
        print(f"Parameter gradient norm percentage analysis plot saved to {save_filename}")
        plt.show()
        print("\n--- Parameter Gradient Norm Percentage Contribution per Epoch ---")
        for i, epoch_data in enumerate(grad_callback_instance.class_grads_history):
            print(f"Epoch {i + 1}:")
            for cls_idx_print in range(classes):
                percentage = epoch_data.get(f'class_{cls_idx_print}_param_grad_norm_percentage', 0.0)
                print(f"  Class {cls_idx_print}: {percentage:.2f}%")
            print("-" * 20)
    else:
        print("No gradient history to plot or 'all' callback instance not found, or history is empty.")
    # --- END OF MODIFIED GRADIENT VISUALIZATION ---


