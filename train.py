import numpy as np
import torch
from torch.utils.data import DataLoader
import os
from torchvision import transforms,datasets
from CNN import CNN
import torch.nn.functional as F

file_path = './data'
train = 'train'
transforms = transforms.Compose(
[
transforms.Resize([1024,1024]),
transforms.ToTensor(),
#transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            #std=[0.229, 0.224, 0.225])
])

batch_size = 1
train_data = datasets.ImageFolder(os.path.join(file_path,train), transforms)
train_loader = DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)
model_path = 'checkpoint/best_parameters.pth' #保存模型参数的路径
lr = 1e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #确认模型所用的设备是gpu还是cpu
model = CNN().to(device) #调用网络结构并将其加载在device上
optimizer = torch.optim.Adam(model.parameters(),lr=lr) #定义优化器，使用Adam

def train(model,device,train_loader,optimizer,epoch):
    best_loss = np.Inf
    model.train()
    for idx,(t_data,t_target) in enumerate(train_loader):
        t_data,t_target = t_data.to(device),t_target.to(device)
        pred = model(t_data)
        loss = F.nll_loss(pred,t_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if idx % 25 == 0:
            print("epoch:{:05d},iteration:{:05d},loss:{:.4f}".format(epoch, idx, loss.item()))
            if loss.item() < best_loss:
                best_loss = loss.item()
                print("saving model ...")
                torch.save(model.state_dict(),model_path)

num_epoch = 10
for epoch in range(num_epoch):
    train(model,device,train_loader,optimizer,epoch)



