import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub


class CNN(nn.Module):
    def __init__(self, class_dim=2):
        super(CNN, self).__init__()

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        #self.conv1 = nn.Conv2d(1, 16, kernel_size=7, padding=3, stride=1, dilation=1)
        self.depthwise_conv1 = nn.Conv2d(1, 1 * 16, kernel_size=7, padding=3, stride=1, dilation=1, groups=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 16, kernel_size=7, padding=3, stride=1, dilation=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=(5, 5), stride=(5, 5))
        self.dropout1 = nn.Dropout(0.3)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=7, padding=3, stride=1, dilation=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=(4, 10), stride=(4, 10))
        self.dropout2 = nn.Dropout(0.3)

        self.fc1 = nn.Linear(32 * 2 * 1, 100)
        self.dropout3 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(100, class_dim)

    def forward(self, x):

        x = self.quant(x)

        #x = self.conv1(x)
        x = self.depthwise_conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        # flatten
        x = x.view(x.size(0), -1)

        # full connect
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        x = F.softmax(x,dim=1)

        x = self.dequant(x)
        return x



if __name__ == "__main__":

    input_shape = (1, 40, 51)
    batch_size = 16

    random_input = torch.randn(batch_size, *input_shape)

    model = CNN(class_dim=2)

    # test
    output = model(random_input)
    print("Output shape:", output.shape)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total number of parameters: {num_params}')
