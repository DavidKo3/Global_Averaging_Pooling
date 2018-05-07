import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import shutil
import torch.optim as optim
import model as model
from torch.autograd import Variable

import time
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--arch', type=str, default='vgg16')
parser.add_argument('--lr', type=float, default=1e-03)
config = parser.parse_args()


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.,5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def imshow(img):
    img = img /2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))

# print labels

print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

if torch.cuda.is_available():
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


def save_checkpoint(state, is_best, epoch, filename='_checkpoint.pth.tar'):
    check_model_file_name= './'+str(epoch)+filename
    torch.save(state, check_model_file_name)
    if is_best:
        shutil.copyfile(check_model_file_name, './'+str(epoch)+'_model_best.pth.tar')

#
# net = model.GAG_Net().cuda()
# # net = model.Net().cuda()
#
# net.weight_init(net.parameters())

# ConvNet as fixed feature extractor
model_conv= torchvision.models.vgg16(pretrained=True)
fine_tune_model = model.FineTuneModel(model_conv, config.arch, 10).cuda()
# print("==============vgg 16 model structure========================/")
# print(fine_tune_model.features)
# print("="*20)
# print(model_conv)

fine_tune_model.weight_init(fine_tune_model.classifier_1.parameters())
fine_tune_model.weight_init(fine_tune_model.classifier_2.parameters())

def adjust_lr(optimizer,epoch, init_lr):
    lr=0
    if epoch ==0:
        lr = init_lr
    else:
        lr = init_lr*(0.5**(epoch//30))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

init_lr = config.lr


params = list(fine_tune_model.classifier_1.parameters())+list(fine_tune_model.classifier_2.parameters())
# print(params)
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(params, lr=init_lr, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer = optim.Adam(params, lr=init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0005)
# optimizer_adam = optim.Adam(net.parameters(), lr=1e-03, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0005 )



adaptive_lr = 0.0

best_acc = 0.0
for epoch in range(300):  # loop over the dataset multiple times

    start_time = time.time()

    adaptive_lr = adjust_lr(optimizer, epoch, init_lr)
    print("epoch : %d , adaptive_lr : %0.5f" % (epoch + 1, adaptive_lr))
    # if epoch ==0:
    #     print("epoch : %d , init_lr : %0.5f" % (epoch + 1, init_lr))
    #     adaptive_lr = adjust_lr(optimizer, epoch, init_lr)
    # else:
    #     print("epoch : %d , adaptive_lr : %0.5f" % (epoch+1, adaptive_lr))
    #     adaptive_lr = adjust_lr(optimizer, epoch, init_lr)
    running_loss = 0.0
    epoch_loss=0.0
    # epoch_test_acc = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = fine_tune_model(inputs)
        # print(outputs)
        # print(outputs[0].shape)
        loss = criterion(outputs, labels)

        # print(loss)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data.cpu().numpy()
        epoch_loss  += running_loss
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss/ 2000))
            running_loss = 0.0

    epoch_loss = epoch_loss/len(trainloader)
    if epoch == 0:
       best_acc = epoch_loss

    print("epoch : %d, epoch_loss : %.5f , best_Acc = %.5f/" % (epoch +1, epoch_loss, best_acc))
    if ((epoch+1) % 50) == 0 and best_acc > epoch_loss:
        best_acc = epoch_loss
        is_best = best_acc
        save_checkpoint({'epoch': epoch + 1, 'arch': config.arch, 'state_dict': fine_tune_model.state_dict(), 'best_acc': best_acc}, is_best, epoch+1)
        elapsed_time = time.time()- start_time
        print("elapsed_time for check best acc :  ", elapsed_time)
print('Finished Training')

