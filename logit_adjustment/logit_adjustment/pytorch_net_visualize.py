from torch import nn
import torch
# from torchinfo import summary
from torchsummary import summary
class Res0(nn.Module):
    def __init__(self):
        super(Res0, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=380, kernel_size=(1, 3), padding=(0, 2)),
                                   nn.BatchNorm2d(380),
                                   nn.ReLU()
                                   # nn.MaxPool2d((1, 2))
                                   )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=380, out_channels=380, kernel_size=(2, 3), padding=(0, 2)),
            nn.BatchNorm2d(380),
            nn.ReLU()
            # nn.MaxPool2d((1, 2))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=380, out_channels=80, kernel_size=(1, 3), padding=(0, 2)),
            nn.BatchNorm2d(80),
            nn.ReLU(),
            # nn.MaxPool2d((1, 2))
        )
        self.rnn1 = nn.LSTM(input_size=80, hidden_size=100, num_layers=1, batch_first=True)
        self.rnn2 = nn.LSTM(input_size=100, hidden_size=50, num_layers=1, batch_first=True)
        self.batch = nn.BatchNorm1d(50)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.reshape(-1, 80, 134)
        x = torch.transpose(x, dim0=1, dim1=2)
        x, (hn, cn) = self.rnn1(x)
        x, (hn1, cn1) = self.rnn2(x)
        x = x[:, -1, :]
        x = self.batch(x)
        x = x.reshape(-1, 50, 1)

        return x


class Res1(nn.Module):
    def __init__(self):
        super(Res1, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=380, kernel_size=(1, 3), padding=(0, 2)),
                                   nn.BatchNorm2d(380),
                                   nn.ReLU()
                                   # nn.MaxPool2d((1, 2))
                                   )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=380, out_channels=380, kernel_size=(2, 3), padding=(0, 2)),
            nn.BatchNorm2d(380),
            nn.ReLU()
            # nn.MaxPool2d((1, 2))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=380, out_channels=80, kernel_size=(1, 3), padding=(0, 2)),
            nn.BatchNorm2d(80),
            nn.ReLU(),
            # nn.MaxPool2d((1, 2))
        )
        self.rnn1 = nn.LSTM(input_size=80, hidden_size=100, num_layers=1, batch_first=True)
        self.rnn2 = nn.LSTM(input_size=100, hidden_size=50, num_layers=1, batch_first=True)
        self.batch = nn.BatchNorm1d(50)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.reshape(-1, 80, 134)
        x = torch.transpose(x, dim0=1, dim1=2)
        x, (hn, cn) = self.rnn1(x)
        x, (hn1, cn1) = self.rnn2(x)
        # attn_output = self.attention_net(x)
        x = x[:, -1, :]
        x = self.batch(x)
        x = x.reshape(-1, 50, 1)

        return x


class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        self.model0 = Res0()
        self.model1 = Res1()
        self.dropout = nn.Dropout(0.1)
        self.linear1 = nn.Linear(2500, 11)
        self.softmax = nn.Softmax(dim=1)
        self.relu = torch.nn.ReLU()
        self.batch = nn.BatchNorm1d(2500)

    def forward(self, x1, x2):
        x1 = self.model0(x1)
        x2 = self.model1(x2)
        x = torch.bmm(x1, torch.transpose(x2, 1, 2)).view(x1.size(0), -1)
        x = self.batch(x)
        x = self.linear1(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        # x = self.linear2(x)
        x = self.softmax(x)
        return x


def weigth_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight.data, gain=torch.nn.init.calculate_gain('relu'))
    elif isinstance(m, torch.nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.fill_(1e-4)
    elif isinstance(m, torch.nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

net = CNN_LSTM()
print(summary(net, (1, 127, 1000), device='cpu'))
