import torch
import data_handler
import model
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import pathlib
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--cont_src_train',default=False,type=bool,help='Whether to continue source training')
parser.add_argument('--epoch_source_start',default=False,type=bool,help='Epoch number to start source training from')
parser.add_argument('--epoch_target_start',default=False,type=bool,help='Whether to continue source training')
parser.add_argument('--num_epochs_source',default=2,type=bool,help='Num of epochs to train source encoder for')
parser.add_argument('--num_epochs_target',default=2,type=bool,help='Num of epochs to train target encoder for')

parser.add_argument('--disc_steps_per_epoch',default=2,type=bool,help='Num of discriminator training steps per epoch of adda')
parser.add_argument('--tar_steps_per_epoch',default=2,type=bool,help='Num of target encoder training steps per epoch of adda')


parser.add_argument('--cont_tar_train',default=False,type=bool,help='Whether to continue target training')
parser.add_argument('--validation_step_source',default=5,type=int,help='Number of epochs after which to do validation in source training')
parser.add_argument('--validation_step_target',default=5,type=int,help='Number of epochs after which to do validation in target training')

parser.add_argument('--save_model_epochs',default=False,type=int,help='No of epochs after which to save model checkpoints')





args = parser.parse_args()


try_no = 1
log_dir = "logs"+str(try_no)
checkpoint_dir = os.path.join("/home/deku/data_adda/checkpoints/",str(try_no))

pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True) 
pathlib.Path(checkpoint_dir).mkdir(parents=True, exist_ok=True) 


writer = SummaryWriter(log_dir=log_dir)



mnist_trainloader = data_handler.get_dataloader_mnist_train(batch_size=4)
mnist_testloader = data_handler.get_dataloader_mnist_test(batch_size=4)
usps_trainloader = data_handler.get_dataloader_usps_train(batch_size=4)
usps_testloader = data_handler.get_dataloader_usps_test(batch_size=4)
mnist_usps_trainloader = data_handler.get_dataloader_mnist_usps_train(batch_size=4)


num_epochs_source_train = 0
validation_step_source = 5

source_encoder = model.LeNet_Enocder()
classifier = model.Classifier()
target_encoder = model.LeNet_Enocder()
discriminator = model.Discrminator()

source_optimizer = optim.Adam([{'params':source_encoder.parameters()},{'params':classifier.parameters()}])

loss_cross_entropy = nn.CrossEntropyLoss()
loss_bce = nn.BCELoss()

source_loss_training = []
source_loss_test = []

source_encoder = source_encoder.to('cuda')
classifier = classifier.to('cuda')
target_encoder = target_encoder.to('cuda')
discriminator = discriminator.to('cuda')

source_encoder_model_path_latest = checkpoint_dir + "source_enocder_latest.pt"
classifier_model_path_latest   = checkpoint_dir + "classifier_latest.pt"
target_encoder_model_path_latest = checkpoint_dir + "target_enocder_latest.pt"
discriminator_model_path_latest = checkpoint_dir + "discriminator_latest.pt"

if args.cont_src_train:
    source_encoder.load_state_dict(torch.load(source_encoder_model_path_latest))
    classifier.load_state_dict(torch.load(classifier_model_path_latest))




for epoch in range(args.epoch_source_start,args.num_epochs_source):


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

    writer.add_scalar('Source Training Loss', training_loss_epoch, epoch)

    torch.save(source_encoder.state_dict(),source_encoder_model_path_latest)
    torch.save(classifier.state_dict(),classifier_model_path_latest)

    with torch.no_grad() :

        if epoch%(args.validation_step_source) == 0 :
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
            writer.add_scalar('Testing Source Encoder Loss', testing_loss_epoch, epoch)

    if epoch%args.save_model_epochs == 0 :
        encoder_path = os.path.join(checkpoint_dir,str(epoch)+"_source_encoder.pt")
        classifier_path = os.path.join(checkpoint_dir,str(epoch)+"_classifier.pt")
        torch.save(source_encoder.state_dict(),encoder_path)
        torch.save(classifier.state_dict(),classifier_path)
    
    


if args.cont_tar_train :
    target_encoder.load_state_dict(torch.load(target_encoder_model_path_latest))
    discriminator.load_state_dict(torch.load(discriminator_model_path_latest))


num_adda_epochs = 2
num_disc_steps_per_epoch = 1
num_tar_steps_per_epoch = 1

for p in source_encoder.parameters():
    p.requires_grad = False

for p in classifier.parameters():
    p.requires_grad = False


target_optimizer = optim.Adam(params=target_encoder.parameters())
disc_optimizer = optim.Adam(params=discriminator.parameters())

disc_loss = []
target_encoder_loss = []
test_target_loss = []

for epoch in range(args.epoch_target_start,args.num_epochs_target):
    disc_loss_epoch = []
    tar_loss_epoch = []
    for disc_iter in range(0,args.disc_steps_per_epoch):
        disc_log_iter = []

        for i,(data_mnist,data_usps) in enumerate(mnist_usps_trainloader):
            
            mnist_images = data_mnist[0]
            mnist_labels = data_mnist[1]
            usps_images = data_usps[0]
            usps_labels = data_usps[1]
            
            mnist_images = mnist_images.to('cuda')
            mnist_labels = mnist_labels.to('cuda')
            usps_images = usps_images.to('cuda')
            usps_labels = usps_labels.to('cuda')


            mnist_encodings = source_encoder(mnist_images)
            usps_encodings = target_encoder(mnist_images)

            mnist_pred = discriminator(mnist_encodings)
            usps_pred = discriminator(usps_encodings)

            label_mnist = torch.ones(len(mnist_labels),1).cuda()
            label_usps = torch.zeros(len(usps_labels),1).cuda()

            loss_mnist = loss_bce(mnist_pred,label_mnist)
            loss_usps = loss_bce(usps_pred,label_usps)

            loss_disc = loss_mnist + loss_usps

            disc_optimizer.zero_grad()
            loss_disc.backward()
            disc_optimizer.step()
            loss_disc = loss_disc.detach().cpu().numpy()
            disc_log_iter.append(loss_disc)
    
        curr_iter_disc_loss = np.mean(np.asarray(disc_log_iter))
        disc_loss_epoch.append(curr_iter_disc_loss)
    
    curr_epoch_disc_loss = np.mean(np.asarray(disc_loss_epoch))
    disc_loss.append(curr_epoch_disc_loss)

    print("Training Discriminator Loss for epoch : {epoch} = {curr_epoch_disc_loss}".format(epoch = epoch , curr_epoch_disc_loss = curr_epoch_disc_loss))
    writer.add_scalar('Training Discriminator Loss', curr_epoch_disc_loss, epoch)


    for tar_iter in range(0,args.tar_steps_per_epoch):
        tar_loss_iter = []

        for i,(data_mnist,data_usps) in enumerate(mnist_usps_trainloader):
            

            usps_images = data_usps[0]
            usps_labels = data_usps[1]
            

            usps_images = usps_images.to('cuda')
            usps_labels = usps_labels.to('cuda')


           
            usps_encodings = target_encoder(mnist_images)
            usps_pred = discriminator(usps_encodings)

            label_usps = torch.ones(len(usps_labels),1).cuda()

           
            loss_usps = loss_bce(usps_pred,label_usps)

            

            target_optimizer.zero_grad()
            loss_usps.backward()
            target_optimizer.step()

            loss_usps = loss_usps.detach().cpu().numpy()
            tar_loss_iter.append(loss_usps)

        curr_iter_tar_loss = np.mean(np.asarray(tar_loss_iter))
        tar_loss_epoch.append(curr_iter_tar_loss)

    curr_epoch_tar_loss = np.mean(np.asarray(tar_loss_epoch))
    target_encoder_loss.append(curr_epoch_tar_loss)

    torch.save(target_encoder.state_dict(),target_encoder_model_path_latest)
    torch.save(discriminator.state_dict(),discriminator_model_path_latest)


    print("Training Target Encoder Loss for epoch : {epoch} = {curr_epoch_tar_loss}".format(epoch = epoch , curr_epoch_tar_loss = curr_epoch_tar_loss))
    writer.add_scalar('Training Target Encoder Loss', curr_epoch_disc_loss, epoch)




    if epoch%args.save_model_epochs == 0 :
        disc_path = os.path.join(checkpoint_dir,str(epoch)+"_discriminator.pt")
        target_encoder_path = os.path.join(checkpoint_dir,str(epoch)+"_target_encoder.pt")
        torch.save(source_encoder.state_dict(),encoder_path)
        torch.save(classifier.state_dict(),classifier_path)


    if epoch%args.validation_step_target == 0 :
        with torch.no_grad() :
            target_encoder = target_encoder.eval()
            for i,(image,label) in enumerate(usps_testloader):
                image = image.to('cuda')
                label = label.to('cuda')
                target_encodings = target_encoder(image)
                pred = classifier(target_encodings)
                loss  =  loss_cross_entropy(pred,label).cpu().numpy()
                epoch_test_loss.append(loss)
            testing_loss_epoch = np.mean(np.asarray(epoch_test_loss))
            test_target_loss.append(testing_loss_epoch) 
            print("Target Testing Loss for epoch : {epoch} = {testing_loss_epoch}".format(epoch = epoch , testing_loss_epoch = testing_loss_epoch))
            writer.add_scalar('Testing Target Encoder Loss', testing_loss_epoch, epoch)


            






    
    
    






