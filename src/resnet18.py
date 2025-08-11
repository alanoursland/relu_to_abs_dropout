import torch
import torch.nn as nn
import torchvision.models.resnet as resnet


class ReLUDropout(nn.Module):
    def __init__(self, dropout_rate=0.5, inplace=True):
        """
        Applies ReLU followed by standard Dropout.
        - dropout_rate: probability of dropping each element.
        - inplace: if True, apply ReLU in-place.
        """
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.relu(x)
        x = self.dropout(x)
        return x


class ReLU2AbsDropout(nn.Module):
    def __init__(self, dropout_rate=0.05):
        """
        dropout_rate: Fraction of elements that become Abs (i.e., a_i = -1)
        """
        super().__init__()
        self.dropout_rate = dropout_rate

    def forward(self, x):
        if not self.training or self.dropout_rate == 0:
            # Act exactly like ReLU in eval mode or if dropout_rate is 0
            return torch.relu(x)

        # Create dropout-like mask same shape as input (no need to exclude batch)
        mask = torch.rand_like(x)
        a = torch.where(mask < self.dropout_rate, torch.full_like(x, -1.0), torch.zeros_like(x))

        return torch.where(x > 0, x, a * x)


# Custom BasicBlock that supports any activation
class BasicBlockWithActivation(resnet.BasicBlock):
    def __init__(self, *args, activation=nn.ReLU(inplace=True), **kwargs):
        super().__init__(*args, **kwargs)
        self.activation = activation

        # Replace the default ReLU with custom activation
        self.relu = activation

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)  # customizable

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)  # customizable again

        return out


# ResNet18 model for CIFAR-10, customizable activation
class ResNet18_CIFAR10(nn.Module):
    def __init__(self, num_classes=10, activation=nn.ReLU(inplace=True)):
        super().__init__()
        self.inplanes = 64
        block = BasicBlockWithActivation
        layers = [2, 2, 2, 2]

        # Initial conv (adjusted for CIFAR-10)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = activation
        self.layer1 = self._make_layer(block, 64, layers[0], activation=activation)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, activation=activation)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, activation=activation)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, activation=activation)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Weight init (optional)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def _make_layer(self, block, planes, blocks, stride=1, activation=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, activation=activation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, activation=activation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
