import data_handler
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import numpy as np

mnist_trainloader = data_handler.get_dataloader_mnist_train(batch_size=100)
usps_trainloader = data_handler.get_dataloader_usps_train(batch_size=100)




images,_ = next(iter(usps_trainloader))
print(images.size())
save_image(images,'usps.png',nrow=10)