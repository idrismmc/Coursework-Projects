import numpy as np
import pandas as pd
import torch
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def angle(a,b):
    x = np.dot(a,b)/np.linalg.norm(a)/np.linalg.norm(b)
    return math.degrees(np.arccos(np.clip(x,-1,1)))
#reading data and preprocessing
df = pd.read_excel("SFEW.xlsx")
df = df.sample(frac=1).reset_index(drop=True)
df = df.fillna(df.mean())
Y = df.iloc[:,1]
Y = Y - 1
X = df.iloc[:,2:]
for i in range(X.shape[1]):
    X.iloc[:,i]=(X.iloc[:,i]-min(X.iloc[:,i]))/(max(X.iloc[:,i])-min(X.iloc[:,i]))

#splitting into training and testing
train_x, test_x, train_y, test_y = train_test_split(X,Y,test_size=0.3)
train_input = torch.Tensor(train_x.values).float()
train_target = torch.Tensor(train_y.values).long()
train_input.size()

#setting parameters
input_neurons = train_input.size()[1]
hidden_neurons = 15
output_neurons = 7
learning_rate = 0.01
num_epochs = 1000

#Define framework
class TwoLayerNet(torch.nn.Module):

    def __init__(self, n_input, n_hidden, n_output):
        super(TwoLayerNet, self).__init__()
        self.hidden = torch.nn.Linear(n_input, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        h_input = self.hidden(x)
        h_output = torch.sigmoid(h_input)
        y_pred = self.out(h_output)
        return y_pred,h_output

#training    
net = TwoLayerNet(input_neurons, hidden_neurons, output_neurons)
loss_func = torch.nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(net.parameters(), lr=learning_rate)
all_losses = []

for epoch in range(num_epochs):
    Y_pred,h = net(train_input)
    loss = loss_func(Y_pred, train_target)
    all_losses.append(loss.item())
    if epoch % 50 == 0:
        _, predicted = torch.max(Y_pred,1)
        total = predicted.size(0)
        correct = predicted.data.numpy() == train_target.data.numpy()
        print('Epoch [%d/%d] Loss: %.4f  Accuracy: %.2f %%'
              % (epoch + 1, num_epochs, loss.item(), 100 * sum(correct)/total))

    net.zero_grad()

    loss.backward()

    optimiser.step()

plt.figure()
plt.plot(all_losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")

#test set accuracy from original model
X_test = torch.Tensor(test_x.values).float()
Y_test = torch.Tensor(test_y.values).long()
Y_pred_test,h1 = net(X_test)
_, predicted_test = torch.max(Y_pred_test, 1)
total_test = predicted_test.size(0)
correct_test = sum(predicted_test.data.numpy() == Y_test.data.numpy())
print('Testing Accuracy: %.2f %%' % (100 * correct_test / total_test))

#finding similar and complementary pairs using hidden layer neuron activation output
n1 = h.detach().numpy().copy()
for i in range(n1.shape[1]):
    if n1[:,i].all() == 0:
        n1[:,i]-=0.5
    else:
        n1[:,i]=(n1[:,i]-min(n1[:,i]))/(max(n1[:,i])-min(n1[:,i]))
        n1[:,i]-=0.5

sim = []
comp = []
for i in range(n1.shape[1]-1):
    for j in range(i+1,n1.shape[1]):
        temp = []
        ang = angle(n1[:,i],n1[:,j])
        temp.append(ang)
        temp.append(i)
        temp.append(j)
        if ang<30:
            sim.append(temp)
        if ang>150:
            comp.append(temp)
print("similar pairs",len(sim))
print("complimentary pairs",len(comp))

#updating necessary connections based on similarity

wt = net.hidden.weight.clone().detach().requires_grad_(True)
b = net.hidden.bias.clone().detach().requires_grad_(True)
bcopy = b.clone().detach().requires_grad_(True)
wtcopy = wt.clone().detach().requires_grad_(True)
compindex = []
for i in range(len(comp)):
    if comp[i][1] not in compindex:
        compindex.append(comp[i][1])
    if comp[i][2] not in compindex:
        compindex.append(comp[i][2])
for i in compindex:
    wt[i] = 0
    b[i] = 0
for i in range(len(sim)):
    wt[sim[i][2]] += wtcopy[sim[i][1]]
    b[sim[i][2]] += bcopy[sim[i][1]]
for i in range(len(sim)):
    wt[sim[i][1]] = 0
    b[sim[i][1]] = 0
wt1 = net.out.weight.clone().detach().requires_grad_(True)
wtcopy1 = wt1.clone().detach().requires_grad_(True)
for i in compindex:
    wt1[:,i] = 0
for i in range(len(sim)):
    wt1[:,sim[i][2]] +=wtcopy1[:,sim[i][1]]
for i in range(len(sim)):
    wt1[:,sim[i][1]] = 0

#test set accuracy after pruning
net.hidden.weight.data = wt.clone().detach().requires_grad_(True)
net.hidden.bias.data = b.clone().detach().requires_grad_(True)
net.out.weight.data = wt1.clone().detach().requires_grad_(True)
Y_pred_test1,_ = net(X_test)
_, predicted_test = torch.max(Y_pred_test1,1)
total_test = predicted_test.size(0)
correct_test = sum(predicted_test.data.numpy() == Y_test.data.numpy())
print('Testing Accuracy after pruning using behaviour: %.2f %%' % (100 * correct_test / total_test))

#finding similar and complementary pairs using output weight matrix
n2 = wtcopy1.detach().numpy().copy()
for i in range(n2.shape[1]):
    if n2[:,i].all() == 0:
        n2[:,i]-=0.5
    else:
        n2[:,i]=(n2[:,i]-min(n2[:,i]))/(max(n2[:,i])-min(n2[:,i]))
        n2[:,i]-=0.5
        
sim1 = []
comp1 = []
for i in range(n2.shape[1]-1):
    for j in range(i+1,n2.shape[1]):
        temp = []
        ang = angle(n2[:,i],n2[:,j])
        temp.append(ang)
        temp.append(i)
        temp.append(j)
        if ang<30:
            sim1.append(temp)
        if ang>150:
            comp1.append(temp)
print("similar pairs",len(sim1))
print("complimentary pairs",len(comp1))

#updating necessary connections based on similarity
wt_ = wtcopy.clone().detach().requires_grad_(True)
b_ = bcopy.clone().detach().requires_grad_(True)
bcopy_ = b_.clone().detach().requires_grad_(True)
wtcopy_ = wt_.clone().detach().requires_grad_(True)
compindex1 = []

for i in range(len(comp1)):
    if comp1[i][1] not in compindex1:
        compindex1.append(comp1[i][1])
    if comp1[i][2] not in compindex1:
        compindex1.append(comp1[i][2])
for i in compindex1:
    wt_[i] = 0
    b_[i] = 0
for i in range(len(sim1)):
    wt_[sim1[i][2]] += wtcopy_[sim1[i][1]]
    b_[sim1[i][2]] += bcopy_[sim1[i][1]]
for i in range(len(sim1)):
    wt_[sim1[i][1]] = 0
    b_[sim1[i][1]] = 0
    
wt1_ = wtcopy1.clone().detach().requires_grad_(True)
wtcopy1_ = wt1_.clone().detach().requires_grad_(True)

for i in compindex1:
    wt1_[:,i] = 0
for i in range(len(sim1)):
    wt1_[:,sim1[i][2]] +=wtcopy1_[:,sim1[i][1]]
for i in range(len(sim1)):
    wt1_[:,sim1[i][1]] = 0

#testing set accuracy after pruning
net.hidden.weight.data = wt_.clone().detach().requires_grad_(True)
net.hidden.bias.data = b_.clone().detach().requires_grad_(True)
net.out.weight.data = wt1_.clone().detach().requires_grad_(True)
Y_pred_test2,_ = net(X_test)
_, predicted_test = torch.max(Y_pred_test2,1)
total_test = predicted_test.size(0)
correct_test = sum(predicted_test.data.numpy() == Y_test.data.numpy())
print('Testing Accuracy after pruning using weight matrix: %.2f %%' % (100 * correct_test / total_test))

plt.show()
