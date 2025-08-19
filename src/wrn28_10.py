import torch
import torch.nn as nn
from typing import Optional
import math

# Reuse your activation modules if desired (optional import).
# These are defined in resnet18.py in your repo and can be passed in as the
# `activation` argument below. If you don't want this dependency here,
# keep the default nn.ReLU and pass custom activations from experiments.py.
try:
    from resnet18 import ReLU2AbsDropout, ReLUDropout  # noqa: F401
except Exception:
    ReLU2AbsDropout = None  # type: ignore
    ReLUDropout = None  # type: ignore


# -----------------------------
# Pre-Activation Wide ResNet
# -----------------------------

class _PreActBlock(nn.Module):
    """
    Pre-activation residual block used by WideResNet on CIFAR.

    Layout (per conv): BN -> (activation) -> Conv3x3.
    Downsampling via stride on conv1 and a 1x1 shortcut.

    If `dropout_between > 0`, a Dropout is inserted between conv1 and conv2
    (the classic WRN intra-block dropout). This is optional because your work
    injects regularization at the activation sites instead.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        activation: Optional[nn.Module] = None,
        dropout_between: float = 0.0,
    ):
        super().__init__()
        self.equal_in_out = (in_ch == out_ch)
        self.stride = stride
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.act1 = activation if activation is not None else nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act2 = activation if activation is not None else nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=float(dropout_between)) if dropout_between and dropout_between > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)

        if not self.equal_in_out:
            # preact variant uses BN/Act on the main path; shortcut is linear proj
            self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn1(x)
        out = self.act1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.act2(out)
        out = self.dropout(out)
        out = self.conv2(out)

        return out + self.shortcut(x)


class WideResNet_CIFAR(nn.Module):
    """
    WideResNet for CIFAR-style inputs (32x32).

    depth = 6*N + 4  ->  N blocks per group (here 28 => N=4)
    widen_factor k scales channels: base [16, 32, 64] -> [16*k, 32*k, 64*k]

    Uses pre-activation basic blocks with BN -> Act -> Conv ordering.
    The provided `activation` module is reused at every ReLU site, matching
    your ResNet-18 pattern (one module instance passed through the network).
    """

    def __init__(
        self,
        depth: int = 28,
        widen_factor: int = 10,
        num_classes: int = 100,
        activation: Optional[nn.Module] = None,
        dropout_between: float = 0.0,
    ):
        super().__init__()
        assert (depth - 4) % 6 == 0, "WRN depth should be 6*N+4"
        n = (depth - 4) // 6  # blocks per group
        k = widen_factor

        widths = [16, 16 * k, 32 * k, 64 * k]

        # Stem
        self.conv1 = nn.Conv2d(3, widths[0], kernel_size=3, stride=1, padding=1, bias=False)

        # Groups
        self.layer1 = self._make_group(widths[0], widths[1], n, stride=1, activation=activation, dropout_between=dropout_between)
        self.layer2 = self._make_group(widths[1], widths[2], n, stride=2, activation=activation, dropout_between=dropout_between)
        self.layer3 = self._make_group(widths[2], widths[3], n, stride=2, activation=activation, dropout_between=dropout_between)

        # Head
        self.bn = nn.BatchNorm2d(widths[3])
        self.relu = activation if activation is not None else nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(widths[3], num_classes)

        self._init_weights()

    def _make_group(
        self,
        in_ch: int,
        out_ch: int,
        num_blocks: int,
        stride: int,
        activation: Optional[nn.Module],
        dropout_between: float,
    ) -> nn.Sequential:
        blocks = []
        for i in range(num_blocks):
            s = stride if i == 0 else 1
            blocks.append(
                _PreActBlock(
                    in_ch=in_ch if i == 0 else out_ch,
                    out_ch=out_ch,
                    stride=s,
                    activation=activation,
                    dropout_between=dropout_between,
                )
            )
        return nn.Sequential(*blocks)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def set_dropout(self, p):
        print("Setting dropout to zero")
        self.relu.set_dropout(p)


# Public builders -------------------------------------------------------------

def wrn28_10_cifar100(activation: Optional[nn.Module] = None, dropout_between: float = 0.0) -> WideResNet_CIFAR:
    """WideResNet-28-10 for CIFAR-100 (num_classes=100).

    Args:
        activation: Module applied at *every* pre-activation site (e.g., nn.ReLU(),
            ReLUDropout(p), or your ReLU2AbsDropout(p)). If None, uses nn.ReLU.
        dropout_between: Classic WRN intra-block dropout probability between conv1
            and conv2 in each block. Leave at 0.0 if you only want activation-site
            regularization.
    """
    return WideResNet_CIFAR(depth=28, widen_factor=10, num_classes=100, activation=activation, dropout_between=dropout_between)


def wrn28_10_cifar10(activation: Optional[nn.Module] = None, dropout_between: float = 0.0) -> WideResNet_CIFAR:
    """WideResNet-28-10 for CIFAR-10 (num_classes=10)."""
    return WideResNet_CIFAR(depth=28, widen_factor=10, num_classes=10, activation=activation, dropout_between=dropout_between)


# Optional quick smoke test ---------------------------------------------------
if __name__ == "__main__":
    import math

    model = wrn28_10_cifar100(activation=nn.ReLU(inplace=True))
    x = torch.randn(2, 3, 32, 32)
    logits = model(x)
    print("OK:", logits.shape)
