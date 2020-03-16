import torch
import data_handler
import model
import torch.nn as nn
import numpy as np
import torch.optim as optim

mnist_trainloader = data_handler.get_dataloader_mnist_train(batch_size=4)
mnist_testloader = data_handler.get_dataloader_mnist_test(batch_size=4)
# usps_trainloader = data_handler.get_dataloader_usps_train(batch_size=8)
# usps_testloader = data_handler.get_dataloader_usps_test(batch_size=8)
# svhn_trainloader = data_handler.get_dataloader_svhn_train(batch_size=8)
# svhn_testloader = data_handler.get_dataloader_svhn_test(batch_size=8)


num_epochs_source_train = 10
validation_step_source = 5

source_encoder = model.LeNet_Enocder()
classifier = model.Classifier()
target_encoder = model.LeNet_Enocder()
discriminator = model.Discrminator()

source_optimizer = optim.Adam([{'params':source_encoder.parameters()},{'params':classifier.parameters()}])

loss_cross_entropy = nn.CrossEntropyLoss()

source_loss_training = []
source_loss_test = []

source_encoder = source_encoder.to('cuda')
classifier = classifier.to('cuda')

for epoch in range(num_epochs_source_train):

    

    epoch_train_loss = []
    epoch_test_loss = []

    source_encoder.train()
    classifier.train()

    for i, (image,label) in enumerate(mnist_trainloader):
        image = image.to('cuda')
        label = label.to('cuda')
        source_encodings = source_encoder(image)
        source_pred = classifier(source_encodings)
        loss  =  loss_cross_entropy(source_pred,label)

        source_optimizer.zero_grad()
        loss.backward()
        loss = loss.detach().cpu().numpy()
        epoch_train_loss.append(loss)
        source_optimizer.step()


    training_loss_epoch = np.mean(np.asarray(epoch_train_loss))
    source_loss_training.append(training_loss_epoch)

    print("Training Loss for epoch : {epoch} = {training_loss_epoch}".format(epoch = epoch , training_loss_epoch = training_loss_epoch))


    with torch.no_grad() :

        if epoch%(validation_step_source) == 0 :
            source_encoder.eval()
            classifier.eval()
            for i,(image,label) in enumerate(mnist_testloader):
                image = image.to('cuda')
                label = label.to('cuda')
                source_encodings = source_encoder(image)
                source_pred = classifier(source_encodings)
                loss  =  loss_cross_entropy(source_pred,label).cpu().numpy()
                epoch_test_loss.append(loss)
            testing_loss_epoch = np.mean(np.asarray(epoch_test_loss))
            source_loss_test.append(testing_loss_epoch) 
            print("Testing Loss for epoch : {epoch} = {testing_loss_epoch}".format(epoch = epoch , testing_loss_epoch = testing_loss_epoch))

    
    

   
    
    
    






