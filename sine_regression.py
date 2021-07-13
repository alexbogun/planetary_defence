import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#Разные активации? Дропоут? Разное количество нейронов и слоев? weight decay? learning rate?

torch.manual_seed(1)
#Reading params
labels = pd.read_csv('./data/parameters.csv', delimiter=',', header=None)
print(labels)
#Define regression model
class net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 10) 
        self.fc2 = nn.Linear(10, 4)
        self.fc3 = nn.Linear(4, 1) 
        self.fc4 = nn.Linear(1, 1) 
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sin(x)
        x = self.fc2(x)
        x = torch.sin(x)
        x = self.fc3(x)
        #x = self.dropout(x)
        x = self.fc4(x)
        #x = torch.sigmoid(x)
        return x

#Actual
if True:
    class classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(142, 90) 
            self.fc2 = nn.Linear(90, 30) 
            self.fc3 = nn.Linear(30, 10) 
            self.fc4 = nn.Linear(10, 3)
            #self.fc3 = nn.Linear(1, 1) 
            self.m = torch.nn.ReLU()

        def forward(self, x):
            x = self.fc1(x)
            x = self.m(x)
            x = self.fc2(x)
            x = self.m(x)
            x = self.fc3(x)
            x = self.m(x)
            x = self.fc4(x)
            x = self.m(x)
            #x = self.fc3(x)
            #x = torch.sigmoid(x)
            return x

    data = []
    #Reading data
    for i in range(1,301):
        print('Light curve:',i,'\n')
        if True:
            if i<10:
                itr = "00"+str(i)
            elif i<100:
                itr = "0"+str(i)
            else:
                itr = str(i)
            fname = './data/lcvold'+itr+'.dat'
            s1 = pd.read_csv(fname, delimiter=',', header=None)
            fname = './data/lcvnew'+itr+'.dat'
            s2 = pd.read_csv(fname, delimiter=',', header=None)

            # Old light curve data
            x1 = []
            for xi in s1[0]:
                x1.append(float(xi))
            y1 = []
            for yi in s1[1]:
                y1.append(float(yi))
            x1 = torch.reshape(torch.tensor(x1), (len(x1),1))
            y1 = torch.reshape(torch.tensor(y1), (len(y1),1))
            #print(x1.size(), y1.size())

            # New light curve data
            x2 = []
            for xi in s2[0]:
                x2.append(float(xi))
            y2 = []
            for yi in s2[1]:
                y2.append(float(yi))
            x2 = torch.reshape(torch.tensor(x2), (len(x2),1))
            y2 = torch.reshape(torch.tensor(y2), (len(y2),1))

        #Training
        if True:
            n_iters = 1000
            #Training for old data
            model1 = net()
            optimizer = torch.optim.Adam(model1.parameters(), lr=0.001) # fallback optimizer
            criterion = torch.nn.MSELoss()  
            for epoch in range(n_iters):
                model1.train()
                optimizer.zero_grad()
                
                out = model1(x1)

                loss = criterion(out, y1)
                loss.backward()

                optimizer.step()
                if epoch%100==0:
                    print('Epoch:',epoch,' loss:',loss.item())
            out = model1(x1)
            weights1 = []
            for param in model1.parameters():
                weights1.append(param.detach().flatten())
            weights1 = torch.cat(weights1)

            #Training for new data
            model2 = net()
            optimizer = torch.optim.Adam(model2.parameters(), lr=0.001) # fallback optimizer
            criterion = torch.nn.MSELoss()  
            for epoch in range(n_iters):
                model2.train()
                optimizer.zero_grad()
                
                out = model2(x1)

                loss = criterion(out, y1)
                loss.backward()

                optimizer.step()
                if epoch%100==0:
                    print('Epoch:',epoch,' loss:',loss.item())
            out = model2(x1)
            weights2 = []
            for param in model2.parameters():
                weights2.append(param.detach().flatten())
            weights2 = torch.cat(weights2)

        weights = torch.cat((weights1,weights2))
        data.append(weights)
    data = torch.stack(data)
    x_train = data[:200]
    x_test = data[200:]
    y_train = torch.transpose(torch.stack(
        (torch.tensor(labels[0], dtype=torch.float32),
        torch.tensor(labels[1], dtype=torch.float32),
        torch.tensor(labels[2],dtype=torch.float32))),0,1)
    model = classifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # fallback optimizer
    criterion = torch.nn.MSELoss()  
    print('\n\n\n***Training classifier***\n')
    for epoch in range(10000):
        model.train()
        optimizer.zero_grad()
        
        out = model(x_train)
        loss = criterion(out, y_train)
        loss.backward()

        optimizer.step()
        if epoch%100==0:
            print('Epoch:',epoch,' loss:',loss.item())
    preds = model(x_test).detach().numpy()

    pd.DataFrame(preds).to_csv('./results.csv', header=False, index=False)

#Testing regression
if False:
    i = 1
    if i<10:
        itr = "00"+str(i)
    elif i<100:
        itr = "0"+str(i)
    else:
        itr = str(i)
    fname = './data/lcvold'+itr+'.dat'
    s1 = pd.read_csv(fname, delimiter=',', header=None)
    fname = './data/lcvnew'+itr+'.dat'
    s2 = pd.read_csv(fname, delimiter=',', header=None)

    # Old light curve data
    x1 = []
    for xi in s1[0]:
        x1.append(float(xi))
    y1 = []
    for yi in s1[1]:
        y1.append(float(yi))
    x1 = torch.reshape(torch.tensor(x1), (len(x1),1))
    y1 = torch.reshape(torch.tensor(y1), (len(y1),1))
    #print(x1.size(), y1.size())

    # New light curve data
    x2 = []
    for xi in s2[0]:
        x2.append(float(xi))
    y2 = []
    for yi in s2[1]:
        y2.append(float(yi))
    x2 = torch.reshape(torch.tensor(x2), (len(x2),1))
    y2 = torch.reshape(torch.tensor(y2), (len(y2),1))
    if True:
        n_iters = 100000
        #Training for old data
        model1 = net()
        optimizer = torch.optim.Adam(model1.parameters(), lr=0.001) # fallback optimizer
        criterion = torch.nn.MSELoss()  
        for epoch in range(n_iters):
            model1.train()
            optimizer.zero_grad()
            
            out = model1(x1)

            loss = criterion(out, y1)
            loss.backward()

            optimizer.step()
            if epoch%1000==0:
                print('Epoch:',epoch,' loss:',loss.item())
        out = model1(x1)
        # x = np.array(range(4500))
        # out2 = model1(torch.tensor(x).unsqueeze(1)*1.0)
        
    #torch.save(model.state_dict(), './models/model'+i+'.pt')
    plt.scatter(x1.numpy(), y1.numpy(), label='true')
    plt.scatter(x1.numpy(), out.detach().numpy(), label='pred')
    #plt.plot(x, out2.detach().numpy(), label='line')
    plt.legend()
    plt.show()
