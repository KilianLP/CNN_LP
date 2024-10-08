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


class VectorQuantizer(nn.Module):
  def __init__(self,emb_dim,n_emb,beta=0.25):
    super().__init__()

    self.embeddings = nn.Embedding(n_emb,emb_dim)
    self.embeddings.weight.data.uniform_(-1.0 / n_emb, 1.0 / n_emb)
    self.beta = beta

  def forward(self,x):

    b,c,h,w = x.size()
    flatten_x = x.permute(0,2,3,1).contiguous()
    flatten_x = flatten_x.reshape(b,h*w,c)

    dist = (flatten_x.unsqueeze(2) - self.embeddings.weight.unsqueeze(0).unsqueeze(0))**2
    dist = dist.sum(dim = -1)
    idx = dist.argmin(dim = -1)
    emb_x = self.embeddings(idx)
    emb_x = emb_x.permute(0,2,1)
    emb_x = emb_x.reshape(b,c,h,w)

    loss = ((x - emb_x.detach())**2).mean() + self.beta * ((x.detach() - emb_x)**2).mean()

    emb_x = emb_x.detach() + x - x.detach()

    return x, loss



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










