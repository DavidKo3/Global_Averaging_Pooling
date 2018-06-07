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


def imshow(img):
    img = img /2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))




parser = argparse.ArgumentParser()
parser.add_argument('--arch', type=str, default='vgg16')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=64)
config = parser.parse_args()

if __name__ == '__main__':
    def adjust_lr(optimizer, epoch, init_lr):
        lr=init_lr

        if (epoch+1) % 25 == 0:
            # lr = init_lr*(0.5**(epoch//10))
            lr = lr * (0.9)


        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(lr)
        return lr


    def save_checkpoint(state, is_best, epoch, filename='_checkpoint.pth.tar'):
        check_model_file_name = './' + str(epoch) + filename
        torch.save(state, check_model_file_name)
        if is_best:
            shutil.copyfile(check_model_file_name, './' + str(epoch) + '_model_best.pth.tar')


    transform = transforms.Compose(
        [transforms.Resize(128), # vgg16 224
         transforms.ToTensor(),
         transforms.Normalize((0., 5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=1)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size, shuffle=False, num_workers=1)

    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))

    # print labels

    # print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    if torch.cuda.is_available():
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    #
    # net = model.GAG_Net().cuda()
    # net = model.Net().cuda()

    # net.weight_init(net.parameters())
    # net = model.CNN().cuda()
    # net.weight_init_cnn()
    #
    net = model.CNN().cuda()
    net.weight_init_cnn()

    # ConvNet as fixed feature extractor
    # model_conv = torchvision.models.vgg16(pretrained=True)
    #
    # for p in model_conv.parameters():
    #     p.requires_grad = False
    #
    # fine_tune_model = model.FineTuneModel(model_conv, config.arch, 10, 1).cuda()
    # # # print("==============vgg 16 model structure========================/")
    # # # print(fine_tune_model.features)
    # # print(fine_tune_model)
    # #
    # #
    # fine_tune_model.weight_init(fine_tune_model.features[-1].parameters())
    # fine_tune_model.weight_init(fine_tune_model.features[-2].parameters())
    # print(fine_tune_model.features[-1])
    # print(fine_tune_model.features[-2])
    # fine_tune_model.weight_init(fine_tune_model.classifier.parameters())


    init_lr = config.lr

    # params = list(fine_tune_model.features[-1].parameters())+list(fine_tune_model.features[-2].parameters())+list(fine_tune_model.classifier.parameters())

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(params, lr=init_lr, momentum=0.9, weight_decay=0.0005, nesterov=True)
    # optimizer = optim.Adam(params, lr=init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0005)
    optimizer = optim.Adam(net.parameters(), lr=init_lr , betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0005 )

    adaptive_lr = 0.0
    best_acc = 9999

    for epoch in range(1000):  # loop over the dataset multiple times
        start_time = time.time()

        adaptive_lr = adjust_lr(optimizer, epoch, init_lr)
        init_lr= adaptive_lr
        print("epoch : %d , adaptive_lr : %0.5f" % (epoch + 1, adaptive_lr))

        running_loss = 0
        epoch_loss = 0

        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            # print(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            # running_loss += loss.data.cpu().numpy()
            running_loss += loss.data[0]
            epoch_loss += loss.data[0]
            # epoch_total_loss += running_loss
            if i % config.batch_size == config.batch_size-1:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss/config.batch_size))
                running_loss = 0.0

        avg_epoch_loss = epoch_loss/len(trainloader)

        if ((epoch+1) % 10) == 0 and best_acc > avg_epoch_loss:

            best_acc = avg_epoch_loss
            is_best = best_acc
            print("epoch : %d, avg_epoch_loss : %.3f , best_Acc = %.3f" % (epoch + 1, avg_epoch_loss, best_acc))
            save_checkpoint({'epoch': epoch + 1, 'arch': config.arch, 'state_dict': net.state_dict(), 'best_acc': best_acc}, is_best, epoch+1)
            elapsed_time = time.time()- start_time
            print("elapsed_time for check best acc :  ", elapsed_time)
    print('Finished Training')

