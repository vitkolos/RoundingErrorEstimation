import torch
import torch.optim as optim
import torch.nn as nn

import tqdm

from dataset import create_dataset
from network import SimpleNet, DenseNet, SmallDenseNet, SmallConvNet

#BATCH_SIZE = 512 #dense mnist
BATCH_SIZE = 1024

# load data
train_data, val_data = create_dataset(train=True, batch_size=BATCH_SIZE)

# create network 
# net = SmallConvNet()#.cuda()
net = SmallDenseNet()#.cuda()
print(net)

# prepare training 
criterion = nn.CrossEntropyLoss()
lr = 0.0001 
#optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0, weight_decay=0.0000)
optimizer = optim.Adam(net.parameters(), weight_decay=0.000)

for epoch in range(1000):  # loop over the dataset multiple times

    running_loss = 0.0
    correct = 0
    num = 0
    i = 0
    net.train()
    for data in tqdm.tqdm(train_data):
        i += 1
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # inputs = inputs.cuda()
        # labels = labels.cuda()
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        _, predicted = torch.max(outputs.data, 1)
        num += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        loss.backward()
        optimizer.step()


        # print statistics
        running_loss += loss.item()
        #        if i % 10 == 9:    # print every 2000 mini-batches
#    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / i:.3f} acc: {100*correct/num:.3f}')

    with torch.no_grad():
        net.eval()
        num, correct = 0, 0
        running_loss = 0
        i = 0
        for inputs, labels in tqdm.tqdm(val_data):
            # inputs, labels = inputs.cuda(), labels.cuda()
            outputs = net(inputs)

            loss = criterion(outputs, labels)        
            _, predicted = torch.max(outputs.data, 1)
            num += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()
            i += 1
    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / i:.3f} vall acc: {100*correct/num:.3f}')

        
    running_loss = 0.0

    if epoch % 5 == 0:
        end = input("End? yes/no")
        if end  == "yes":
            break

#    if epoch % 25 == 24:
#        lr *= 0.5
#        for g in optimizer.param_groups:
#            g['lr'] = lr
#        print("New learning rate: ", lr)


print('Finished Training')

# PATH = './mnist_conv_net.pt'
PATH = './mnist_dense_net.pt'
torch.save(net.state_dict(), PATH)    
print("Network saved.")
