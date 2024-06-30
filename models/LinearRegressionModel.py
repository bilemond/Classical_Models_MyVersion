import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter


# 定义模型
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # 输入和输出都是1维

    def forward(self, x):
        return self.linear(x)


if __name__ == '__main__':
    # 准备数据
    x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                        [9.779], [6.182], [7.59], [2.167],
                        [7.042], [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

    y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                        [3.366], [2.596], [2.53], [1.221],
                        [2.827], [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)

    # 初始化模型
    model = LinearRegressionModel()

    # 损失和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # 初始化SummaryWriter
    writer = SummaryWriter('../runs/linear_regression_experiment')   # step1

    # 训练模型
    num_epochs = 100
    for epoch in range(num_epochs):
        # 转换为tensor
        inputs = x_train
        targets = y_train

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录损失
        writer.add_scalar('Loss/train', loss.item(), epoch)  # step2

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 关闭SummaryWriter
    writer.close()  # tensorboard --logdir=runs/
