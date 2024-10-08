import torch as T
import torch.nn as nn

class EncoderBlock(nn.Module):
  def __init__(self,input_chanel,out_chanel):
    super().__init__()

    self.net = nn.Sequential(
        nn.Conv2d(input_chanel,out_chanel,3,stride = 2, padding = 1),
        nn.BatchNorm2d(out_chanel),
        nn.GELU(),
    )

  def forward (self,x):
    return self.net(x)


class DecoderBlock(nn.Module):
  def __init__(self,input_chanel,out_chanel):
    super().__init__()

    self.net = nn.Sequential(
        nn.ConvTranspose2d(input_chanel,out_chanel,3,stride = 2, padding = 1, output_padding = 1),
        nn.BatchNorm2d(out_chanel),
        nn.GELU(),
        )

  def forward (self,x):
    return self.net(x)


class Encoder(nn.Module):
  def __init__(self,chanels):
    super().__init__()

    self.net = nn.Sequential()
    for c in range(len(chanels)-1):
      self.net.add_module(f"enc_block_{c}",EncoderBlock(chanels[c],chanels[c+1]))

  def forward(self,x):
    return self.net(x)

class Decoder(nn.Module):
  def __init__(self,chanels):
    super().__init__()

    self.net = nn.Sequential()
    for c in range(len(chanels)-1):
      self.net.add_module(f"dec_block_{c}",DecoderBlock(chanels[c],chanels[c+1]))

  def forward(self,x):
    return self.net(x)


class Autoencoder(nn.Module):
  def __init__(self,n_emb):
    super().__init__()

    self.encoder = Encoder([3,64,128,256,512])
    self.vector_quantizer = VectorQuantizer(512,n_emb)
    self.decoder = Decoder([512,256,128,64,3])

    self.tanh = nn.Tanh()

  def forward(self,x):
    x = self.encoder(x)
    x, loss = self.vector_quantizer(x)
    x = self.decoder(x)
    x = self.tanh(x)

    return x, loss










