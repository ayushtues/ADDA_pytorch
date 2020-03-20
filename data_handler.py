import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

transform_mnist = transforms.Compose(
    [transforms.Resize([32,32]),transforms.ToTensor()])

transform_svhn = transforms.Compose([transforms.ToTensor()])     
transform_usps = transforms.Compose([transforms.Resize([32,32]),transforms.ToTensor()])     


mnist_data_train = torchvision.datasets.MNIST('/home/deku/Coding/data_adda/datasets_adda', train=True, transform=transform_mnist, target_transform=None, download=True)
svhn_data_train = torchvision.datasets.SVHN('/home/deku/Coding/data_adda/datasets_adda', split='train', transform=transform_svhn, target_transform=None, download=True)
usps_data_train = torchvision.datasets.USPS('/home/deku/Coding/data_adda/datasets_adda', train=True, transform=transform_usps, target_transform=None, download=True)

mnist_data_test = torchvision.datasets.MNIST('/home/deku/Coding/data_adda/datasets_adda', train=False, transform=transform_mnist, target_transform=None, download=True)
svhn_data_test = torchvision.datasets.SVHN('/home/deku/Coding/data_adda/datasets_adda', split='test', transform=transform_svhn, target_transform=None, download=True)
usps_data_test = torchvision.datasets.USPS('/home/deku/Coding/data_adda/datasets_adda', train=False, transform=transform_usps, target_transform=None, download=True)





class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

mnist_usps_train = ConcatDataset(mnist_data_train,usps_data_train)


def get_dataloader_mnist_usps_train(batch_size):
    print("LEN OF MNIST  :",len(mnist_data_train))
    print("LEN OF MNIST  :",len(usps_data_train))
    print("LEN OF MNIST + USPS dataset :",len(mnist_usps_train))
    dataloader_mnist_usps_train = torch.utils.data.DataLoader(mnist_usps_train,batch_size=batch_size,drop_last=True,shuffle=True)
    return dataloader_mnist_usps_train

def get_dataloader_mnist_train(batch_size):
    dataloader_mnist_train = torch.utils.data.DataLoader(mnist_data_train,batch_size=batch_size,drop_last=True,shuffle=True)
    return dataloader_mnist_train


def get_dataloader_mnist_test(batch_size):
    dataloader_mnist_test = torch.utils.data.DataLoader(mnist_data_test,batch_size=batch_size,drop_last=True,shuffle=True)
    return dataloader_mnist_test


def get_dataloader_svhn_train(batch_size):
    dataloader_svhn_train = torch.utils.data.DataLoader(svhn_data_train,batch_size=batch_size,drop_last=True,shuffle=True)
    return dataloader_svhn_train


def get_dataloader_svhn_test(batch_size):
    dataloader_svhn_test = torch.utils.data.DataLoader(svhn_data_test,batch_size=batch_size,drop_last=True,shuffle=True)
    return dataloader_svhn_test



def get_dataloader_usps_train(batch_size):
    dataloader_usps_train = torch.utils.data.DataLoader(usps_data_train,batch_size=batch_size,drop_last=True,shuffle=True)
    return dataloader_usps_train



def get_dataloader_usps_test(batch_size):
    dataloader_mnist_train = torch.utils.data.DataLoader(usps_data_test,batch_size=batch_size,drop_last=True,shuffle=True)
    return dataloader_mnist_train                
