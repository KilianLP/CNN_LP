import torch.optim as optim
import torch.nn as nn
from models import Autoencoder


device = T.device("cuda" if T.cuda.is_available() else "cpu")

autoencoder = Autoencoder(512)
autoencoder.to(device)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr = 1e-3)

for _ in range(40):
  total_loss = 0
  autoencoder.train()
  
  for batch,_ in dataloader:
    optimizer.zero_grad()
    batch = batch.to(device)
    
    output,loss = autoencoder(batch)
    loss_value = loss_fn(output,batch) + loss
    loss_value.backward()
    
    optimizer.step()
    
    total_loss += loss_value.item()
    
  print("train_loss =",total_loss/len(dataloader))

  autoencoder.eval()
  total_loss = 0
  
  with T.no_grad():
    for batch,_ in dataloader_test:
      batch = batch.to(device)
      output,_ = autoencoder(batch)
      loss_value = loss_fn(output,batch)
      total_loss += loss_value.item()
  print("test_loss =",total_loss/len(dataloader_test))
