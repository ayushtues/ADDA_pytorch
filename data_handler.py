import torch
import torchvision
import torchvision.transforms as transforms

transform_mnist = transforms.Compose(
    [transforms.Resize([32,32]),transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))])

mnist_data_train = torchvision.datasets.MNIST('./datasets', train=True, transform=transform_mnist, target_transform=None, download=True)
svhn_data_train = torchvision.datasets.SVHN('./datasets', split='train', transform=None, target_transform=None, download=True)
usps_data_train = torchvision.datasets.USPS('./datasets', train=True, transform=None, target_transform=None, download=True)

mnist_data_test = torchvision.datasets.MNIST('./datasets', train=False, transform=transform_mnist, target_transform=None, download=True)
svhn_data_test = torchvision.datasets.SVHN('./datasets', split='test', transform=None, target_transform=None, download=True)
usps_data_test = torchvision.datasets.USPS('./datasets', train=False, transform=None, target_transform=None, download=True)






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
