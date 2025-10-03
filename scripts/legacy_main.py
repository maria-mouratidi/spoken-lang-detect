import numpy as np
import torch
import torch.nn as nn
import os as os
import torch.optim as optim 

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print(torch.cuda.get_device_name(0))
else:
    device = 'cpu'
print("Device: {}".format(device))

sampling_rate = 8_000
languages = ["de", "en", "es", "fr", "nl", "pt"]
language_dict = {languages[i]: i for i in range(len(languages))}

X_train, y_train = np.load("dataset/inputs_train_fp16.npy"), np.load(
    "dataset/targets_train_int8.npy"
)
X_test, y_test = np.load("dataset/inputs_test_fp16.npy"), np.load(
    "dataset/targets_test_int8.npy"
)

X_train, X_test = X_train.astype(np.float32), X_test.astype(np.float32)

X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# has to be a dataset class otherwise DataLoader doesnt work
class LanguageDataset():
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y 
    
# Create dataset objects
mean = torch.mean(X_train)
std = torch.std(X_train)

train_dataset = LanguageDataset(X_train, y_train)
test_dataset = LanguageDataset(X_test, y_test)

# Load data using DataLoader with batch size
batch_size = 1
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True) 

class LanguageClassifier(nn.Module):
    def __init__(self):
       
        super(LanguageClassifier, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, kernel_size=50, stride=5)
        #self.bn1 = nn.BatchNorm1d(32)  #batchnorm showed 3%less test accuracy
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1)
        #self.bn2 = nn.BatchNorm1d(64)  
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1)
        #self.bn3 = nn.BatchNorm1d(128)  
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool1d(kernel_size=3, stride=2)
        
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, stride=1)
        #self.bn4 = nn.BatchNorm1d(256)  
        self.relu4 = nn.ReLU()
        self.maxpool4 = nn.MaxPool1d(kernel_size=3, stride=2)
        
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        
        self.relu6 = nn.ReLU()
        self.fc4 = nn.Linear(64, 6)

    def forward(self, x):
        x = self.conv1(x)
        #x = self.bn1(x)  
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        #x = self.bn2(x)  
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        #x = self.bn3(x)  
        x = self.relu3(x)
        x = self.maxpool3(x)

        x = self.conv4(x)
        #x = self.bn4(x)  
        x = self.relu4(x)
        x = self.maxpool4(x)
       
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        
        
        x = self.fc2(x)
        x = self.fc3(x)  
        x = self.relu6(x)
        x = self.fc4(x)

        return x 
    


model = LanguageClassifier() 

# train model
optimizer = optim.Adam(model.parameters(), lr=0.001)
crossentropy_loss = nn.CrossEntropyLoss()
num_epochs = 60
model.to(device)
model.train() 

for epoch in range(num_epochs):
    correct_predictions = 0
    train_acc = 0
    for batch_inputs, batch_targets in train_loader: 
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        batch_inputs = batch_inputs.view(1, 1, 40000)
        optimizer.zero_grad()           
        predictions = model(batch_inputs)  

        accuracy_batch = (torch.max(predictions, dim=-1, keepdim=True)[1].flatten() == batch_targets).sum() / len(batch_targets)
        train_acc += accuracy_batch.item()

        loss = crossentropy_loss(predictions, batch_targets) 
        loss.backward() 
        optimizer.step()
    
    model.eval() 
    test_acc = 0 
    for batch_inputs, batch_targets in test_loader: 
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        batch_inputs = batch_inputs.view(1, 1, 40000)
        
        predictions = model(batch_inputs)
        _, predicted_labels = torch.max(predictions, dim=-1)
        
        accuracy = (predicted_labels == batch_targets).sum() / len(batch_targets)
        test_acc += accuracy.item()
    
    test_acc /= len(test_loader)
    train_acc /= len(train_loader)    
    print(f"epoch: [{epoch}], Training Batch Accuracy: [{train_acc}], Testing Batch Accuracy: [{test_acc}]")

# save model
torch.save(model.state_dict(), "model_test_state14.pt")


