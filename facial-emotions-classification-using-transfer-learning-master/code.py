#Idris Mustafa
#u6733671

from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torchvision import models
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import random
import math
import os
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-0.5,0.5))
detector = MTCNN()

# Preprocessing block - Preprocessing images with face detector, saving images in a new directory named 'data'

'''directory = 'original data'
print('Preprocessing images with face detector')
for j in os.listdir(directory):
    for i in os.listdir(directory+'/'+j):
        img = np.array(Image.open(directory+'/'+j+'/'+i))
        face = detector.detect_faces(img)
        for f in face:
            if f['box'][0]<0:
                f['box'][0]=0
            if f['box'][1]<0:
                f['box'][1]=0
            img1 = img[f['box'][1]:f['box'][1]+f['box'][3],f['box'][0]:f['box'][0]+f['box'][2]]
            img1 = Image.fromarray(img1.astype('uint8'))
            if not os.path.exists('data'+'/'+j):
                os.makedirs('data'+'/'+j)
            img1.save('data'+'/'+j+'/'+i)
print('Preprocessed images saved in data directory')'''

#setting seed
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True
    
batch_size = 32
num_epochs = 65
data_folder='data'

#using ImageLoader to prepare input and targets for the deep learning model
transform = transforms.Compose(
        [transforms.Resize((299,299)),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])
data = ImageFolder(root = data_folder, transform=transform)
print(data.class_to_idx)
train_size = int(0.7 * len(data))
test_size = len(data)-train_size
train_set,test_set = torch.utils.data.random_split(data, [train_size, test_size])
train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set,batch_size=batch_size,shuffle=True)

#Loading alexnet pretrained model
alexnet = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
# print(alexnet)
class Net(nn.Module):
    def __init__(self,alexnet):
        super(Net, self).__init__()
        
	#saving all layers except the last layer
        self.alexnet_features = nn.Sequential(*list(alexnet.children())[:-1])
        self.classifier = nn.Linear(9216,1024)
        self.classifier1 = nn.Linear(1024,512)
        self.classifier2 = nn.Linear(512,7)
        self.dropout = nn.Dropout()
	
	#freezing all layers except the last layer
        for f in self.alexnet_features.parameters():
            f.requires_grad=False
            
    def forward(self,x):
        h0_out=self.alexnet_features(x)
        h0_out = h0_out.view(h0_out.size(0),-1)
        h0_out = self.dropout(h0_out)
        h1_in = self.classifier(h0_out)
        h1_out = torch.sigmoid(h1_in)
        h1_out = self.dropout(h1_out)
        h2_in = self.classifier1(h1_out)
        h2_out = torch.sigmoid(h2_in)
        ypred = self.classifier2(h2_out)
        return ypred,h2_out

#Architecture of our own model implemented for the pruning experiment

#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3,stride=1,padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3,stride=1,padding=1)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=5,stride=1,padding=2)
#         self.conv3_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(128*37*37, 100)
#         self.fc2 = nn.Linear(100,7)

#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2(x), 2))
#         x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
#         x = x.view(x.size(0),-1)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return x

model = Net(alexnet)
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')
    
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

#train model on GPU if available
if torch.cuda.is_available():
    model = model.cuda()

#training block
for num in range(num_epochs):
    train_loss = 0
    model.train()
    total = 0
    correct = 0
    h1 = []
    for batch_idx, (data, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data,target=data.cuda(),target.cuda()
        optimizer.zero_grad()
        output,h = model(data)
        h1.append(h)
        _,pred = torch.max(output.data,1)
        total+=target.size(0)
        correct+=(pred==target).sum().item()
        loss = criterion(output, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    res = torch.cat(h1,dim=0)
    print('Epochs : {}/{} - loss : {} training accuracy - {}'.format(num+1,num_epochs,train_loss/len(train_loader),100*correct/total))


#Function implemented to find the angle between two vectors
#Function takes two input vectors and returns angle in degrees
def find_angle(a,b):
    v1 = a.astype('float')
    v2 = b.astype('float')
    u1 = v1/np.linalg.norm(v1)
    u2 = v2/np.linalg.norm(v2)
    dp = np.dot(u1,u2)
    dp = np.clip(dp,-1,1)
    return math.degrees(np.arccos(dp))

#Function implemented that finds the similar and complementory pairs of neurons as described in the paper
#Angular separation threshold of 30 degrees is set to detect similar and complementary pairs
def find_sim_comp(n1):
    sim = []
    comp = []
    for i in range(n1.shape[1]-1):
        for j in range(i+1,n1.shape[1]):
            temp = []
            ang = find_angle(n1[:,i],n1[:,j])
            temp.append(ang)
            temp.append(i)
            temp.append(j)
            if ang<30:
                sim.append(temp)
            if ang>150:
                comp.append(temp)
    print("similar pairs",len(sim))
    print("complementary pairs",len(comp))
    return sim,comp

#Pruning of neurons by turning off weight connections that goes in and out of the hidden layer
def prune_weights(sim,comp,wt,b,wt1):
    compindex = []
    wtcopy=wt.clone()
    bcopy=b.clone()
    wtcopy1 = wt1.clone()
    for i in range(len(comp)):
        if comp[i][1] not in compindex:
            compindex.append(comp[i][1])
        if comp[i][2] not in compindex:
            compindex.append(comp[i][2])
    for i in compindex:
        wt[i] = 0
        b[i] = 0
        wt1[:,i]=0
    for i in range(len(sim)):
        wt[sim[i][2]] += wtcopy[sim[i][1]]
        b[sim[i][2]] += bcopy[sim[i][1]]
        wt1[:,sim[i][2]] +=wtcopy1[:,sim[i][1]]
    for i in range(len(sim)):
        wt[sim[i][1]] = 0
        b[sim[i][1]] = 0
        wt1[:,sim[i][1]] = 0
        
    c=0
    for i in range(wt.clone().numpy().shape[0]):
        if wt.clone().numpy()[i].all()==0:
            c+=1
    print('Number of neurons pruned - {}/{}'.format(c,len(wt)))
    if torch.cuda.is_available():
        wt,b,wt1=wt.cuda(),b.cuda(),wt1.cuda()
    return wt.clone(),b.clone(),wt1.clone()

#Function implemented to check test set accuracy
#Function takes the model variable and the test set as input and returns prediction accuracy
def accuracy(model,test_loader):
    model.eval()

    with torch.no_grad():
        total = 0
        correct = 0
        for data, target in test_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            output,_ = model(data)
            _,pred = torch.max(output.data , 1)
            total+=target.size(0)
            correct += (pred == target).sum().item()
        
    return 100*correct/total

#copying original weights and biases required for the two pruning process 
originalwt = model.classifier1.weight.clone().data.detach()
originalb = model.classifier1.bias.clone().data.detach()
originalwt1 = model.classifier2.weight.clone().data.detach()

set_seed(10)
print('Test accuracy without pruning -',accuracy(model,test_loader))

#Pruning process using the neuron activation output
n1 = np.array(res.cpu().detach())
#normalizing vector to range -5,5
n1 = scaler.fit_transform(n1)
sim,comp=find_sim_comp(n1)
x,y,z = prune_weights(sim,comp,originalwt.cpu(),originalb.cpu(),originalwt1.cpu())

#saving edited weights to our model
model.classifier1.weight.data = x
model.classifier1.bias.data = y
model.classifier2.weight.data = z
set_seed(10)
print('Test accuracy after pruning using neuron activation output -',accuracy(model,test_loader))

#Pruning process using the static weights on output neurons 

n2 = scaler.fit_transform(originalwt1.cpu().clone().numpy())
sim1,comp1 = find_sim_comp(n2)
x,y,z = prune_weights(sim1,comp1,originalwt.cpu().clone(),originalb.cpu().clone(),originalwt1.cpu().clone())
model.classifier1.weight.data = x
model.classifier1.bias.data = y
model.classifier2.weight.data = z
set_seed(10)
print('Test accuracy after pruning using static weight matrix -',accuracy(model,test_loader))