import torch as T
import torch.nn as nn

class Hirarchical_autoencoder(nn.Module):
  def __init__(self,n_emb):
    super().__init__()
    self.encoder = Encoder([3,64,128,256,512])
    self.vector_quantizer = VectorQuantizer(512,n_emb)
    self.decoder = Decoder([512,256,128,64,3])

    self.encoder_2 = Encoder([512,768,1024])
    self.vector_quantizer_2 = VectorQuantizer(1024,n_emb)
    self.decoder_2 = Decoder([1024,768,512])

    self.tanh = nn.Tanh()

  def forward(self,x):
    y = self.encoder(x)
    z = self.encoder_2(y)
    z, loss_2 = self.vector_quantizer_2(z)
    y_ = self.decoder_2(z)
    y, loss = self.vector_quantizer(y)
    y = y_ + y
    x = self.decoder(y)
    x = self.tanh(x)
    loss = loss + loss_2

    return x,loss
