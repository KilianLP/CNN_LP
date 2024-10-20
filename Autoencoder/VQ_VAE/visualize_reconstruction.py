import random
import numpy as np
import matplotlib.pyplot as plt


images = []
for i in range(5):
    k = random.randrange(1, len(test_dataset))
    image,_= test_dataset.__getitem__(k)
    image = image.permute(1,2,0)
    image = image.numpy()
    images.append(image)

    autoencoder.eval()
    image,_= test_dataset.__getitem__(k)
    image, _= autoencoder(image.unsqueeze(0).to(device))
    image = image.squeeze()
    image = image.cpu()
    image = image.permute(1,2,0)
    image = image.detach().numpy()

    images.append(image)


fig, axs = plt.subplots(5,2 , figsize=(6, 15))


for i, ax in enumerate(axs.flat):
    ax.imshow(images[i])  
    if i % 2 == 0:
        ax.set_title(f'Original Image {i//2+1}') 
    else:
      ax.set_title(f'Reconstructed Image {i//2 + 1}') 
    ax.axis('off')  

plt.tight_layout() 
plt.show()
