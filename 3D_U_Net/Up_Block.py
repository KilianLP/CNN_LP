import torch as T
import torch.nn as nn

class UpBlock(nn.Module):
  def __init__(self,in_channels,hidden_channels,out_channels,kernel_size, padding, scale_factor):
    super().__init__()

    self.conv1 = nn.Conv3d(in_channels,hidden_channels,kernel_size,padding = padding)
    self.batch_norm1 = nn.BatchNorm3d(hidden_channels)
    self.conv2 = nn.Conv3d(hidden_channels,out_channels,kernel_size,padding = padding)
    self.batch_norm2 = nn.BatchNorm3d(out_channels)

    self.relu = nn.ReLU()
    self.upsample = nn.Upsample(scale_factor = scale_factor ,mode='trilinear', align_corners=True)

  def forward(self,x,pooled_x):
    x = self.upsample(x)
    pooled_x = self.resize(pooled_x, x.shape[2:])
    x = T.cat((x,pooled_x),dim = 1)
    x = self.conv1(x)
    x = self.batch_norm1(x)
    x = self.relu(x)
    x = self.conv2(x)
    x = self.batch_norm2(x)
    x = self.relu(x)

    return x

  def resize(self, x, size):
        return F.interpolate(x, size=size, mode='trilinear', align_corners=True)
