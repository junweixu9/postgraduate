import torch

def masked_mse_with_penalty( pred, target: torch.Tensor, mask,  lambda_penalty) :
    M = torch.sum(mask)
    loss_mse = 0
    for i in range(len(pred)) :
        if mask[i] == 1 :
            loss_mse += torch.square(pred[i] - target[i])
    loss_mse = loss_mse/(M+1e-6)

    loss_pen = 0
    for i in range(len(pred)) :
        if mask[i] == 1 :
            if pred[i] > target[i] :
                loss_pen += torch.square(pred[i] - target[i])
    loss_pen = loss_pen/(M+1e-6)

    loss = loss_mse + lambda_penalty * loss_pen

    return loss

if __name__ == '__main__':

    pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
    target = torch.tensor([0.5, 2.5, 3.0, 3.0])
    mask = torch.tensor([1, 1, 0, 1])

    loss = masked_mse_with_penalty(pred, target, mask, lambda_penalty=0.1)
    print("loss", loss)
