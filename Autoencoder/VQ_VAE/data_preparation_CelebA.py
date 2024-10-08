import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,random_split

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = torchvision.datasets.ImageFolder(root="./celeba_data/img_align_celeba", transform=transform)

train_size = int(0.8 * len(dataset))  
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

dataloader_train = DataLoader(train_dataset, batch_size=64, shuffle=True)
dataloader_test = DataLoader(test_dataset, batch_size=64, shuffle=True)
