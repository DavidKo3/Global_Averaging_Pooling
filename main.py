import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

import torch.optim as optim
import model as model
from torch.autograd import Variable
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

#
net = model.GAG_Net().cuda()
# net = model.Net().cuda()

net.weight_init(net.parameters())

# ConvNet as fixed feature extractor
model_conv= torchvision.models.vgg16_bn(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False


print(model_conv)


print(model_conv.features)
# print(model_conv.classifier)

# mod= list(model_conv.features.children())
# gap_layer = []
# gap_layer.append()
# print(mod)

def adjust_lr(optimizer,epoch, init_lr):
    lr=0
    if epoch ==0:
        lr = init_lr
    else:
        lr = init_lr*(0.5**(epoch//30))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=init_lr, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer_adam = optim.Adam(net.parameters(), lr=1e-03, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0005 )



adaptive_lr = 0.0
for epoch in range(300):  # loop over the dataset multiple times


    adaptive_lr = adjust_lr(optimizer, epoch, init_lr)
    print("epoch : %d , adaptive_lr : %0.5f" % (epoch + 1, adaptive_lr))
    # if epoch ==0:
    #     print("epoch : %d , init_lr : %0.5f" % (epoch + 1, init_lr))
    #     adaptive_lr = adjust_lr(optimizer, epoch, init_lr)
    # else:
    #     print("epoch : %d , adaptive_lr : %0.5f" % (epoch+1, adaptive_lr))
    #     adaptive_lr = adjust_lr(optimizer, epoch, init_lr)
    running_loss = 0.0
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
        # print(outputs[0].shape)
        loss = criterion(outputs, labels)

        # print(loss)
        loss.backward()
        optimizer_adam.step()

        # print statistics
        running_loss += loss.data.cpu().numpy()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss/ 2000))
            running_loss = 0.0


print('Finished Training')

