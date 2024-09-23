import torch as T
import torch.nn as nn

class DownBlock(nn.Module):
  def __init__(self,in_channels,hidden_channels,out_channels,kernel_size, padding):
    super().__init__()

    self.conv1 = nn.Conv3d(in_channels,hidden_channels,kernel_size,padding = padding)
    self.batch_norm1 = nn.BatchNorm3d(hidden_channels)
    self.conv2 = nn.Conv3d(hidden_channels,out_channels,kernel_size,padding = padding)
    self.batch_norm2 = nn.BatchNorm3d(out_channels)

    self.relu = nn.ReLU()
    self.pool = nn.MaxPool3d(2,stride = 2)

  def forward(self,x):
    x = self.conv1(x)
    x = self.batch_norm1(x)
    x = self.relu(x)
    x = self.conv2(x)
    x = self.batch_norm2(x)
    x = self.relu(x)
    pooled_x = self.pool(x)

    return x,pooled_x
