import numpy as np
import random

int = np.arange(3)
y_pred = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
y_true = [2, 1, 3]
l = list(zip(y_true, int))
random.shuffle(l)
s1, s2 = list(zip(*l))
print(s2)
yibufen_attack_pred = s1[:2]
yibufen_attack_pred = np.array(yibufen_attack_pred)
print(yibufen_attack_pred)
all_pred_selected = [
    [each_pred[i] for i in s2[:2]]  # Extract specific indices for each row
    for each_pred in y_pred
]
print(all_pred_selected)
yibufen_attack_plt = np.array(all_pred_selected)
print(all_pred_selected)


f = "one_attack_traces"
h = "attack_traces" in f
print(h)
y_pred = np.array(y_pred)
sorted_indices = np.argsort(y_true).tolist()
y = y_pred[sorted_indices]
print(y)
# print(int)
# y_pred = np.random.rand(3)
# l = list(zip(y_pred, int))
# random.shuffle(l)
# s1, s2 = list(zip(*l))
#
# np.set_printoptions(threshold=np.inf)
# print(s2)
