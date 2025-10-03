#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
import re
import torch.nn as nn
from torchview import draw_graph #requires graphviz to be installed in the environment


# In[33]:


accuracy_output = """
epoch: [0], Training Batch Accuracy: [0.25683334870263935], Testing Batch Accuracy: [0.2841666826978326]
tensor(1.5600, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [1], Training Batch Accuracy: [0.2823333506286144], Testing Batch Accuracy: [0.28916668370366094]
tensor(1.6983, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [2], Training Batch Accuracy: [0.29616668455302714], Testing Batch Accuracy: [0.3133333527483046]
tensor(1.6670, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [3], Training Batch Accuracy: [0.31283335233107207], Testing Batch Accuracy: [0.29166668374091387]
tensor(1.5196, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [4], Training Batch Accuracy: [0.33266668692231177], Testing Batch Accuracy: [0.3375000190921128]
tensor(1.2785, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [5], Training Batch Accuracy: [0.34533335419371725], Testing Batch Accuracy: [0.35500002223998306]
tensor(1.4950, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [6], Training Batch Accuracy: [0.360000022072345], Testing Batch Accuracy: [0.3950000233016908]
tensor(1.5746, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [7], Training Batch Accuracy: [0.37816668873652814], Testing Batch Accuracy: [0.3891666910611093]
tensor(1.5680, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [8], Training Batch Accuracy: [0.39400002354756], Testing Batch Accuracy: [0.3950000235810876]
tensor(1.5333, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [9], Training Batch Accuracy: [0.40716669013723733], Testing Batch Accuracy: [0.42000002358108757]
tensor(1.4776, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [10], Training Batch Accuracy: [0.4408333585038781], Testing Batch Accuracy: [0.4233333572745323]
tensor(0.9159, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [11], Training Batch Accuracy: [0.44500002432614566], Testing Batch Accuracy: [0.4466666908934712]
tensor(1.6035, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [12], Training Batch Accuracy: [0.4681666916050017], Testing Batch Accuracy: [0.4816666927188635]
tensor(0.7651, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [13], Training Batch Accuracy: [0.4913333596661687], Testing Batch Accuracy: [0.46166669186204673]
tensor(1.5361, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [14], Training Batch Accuracy: [0.5045000259950757], Testing Batch Accuracy: [0.5083333579823375]
tensor(1.1306, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [15], Training Batch Accuracy: [0.5205000263452529], Testing Batch Accuracy: [0.52083335891366]
tensor(0.9658, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [16], Training Batch Accuracy: [0.5323333591595292], Testing Batch Accuracy: [0.5191666923463345]
tensor(0.9045, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [17], Training Batch Accuracy: [0.5440000271052122], Testing Batch Accuracy: [0.5258333593606949]
tensor(0.6832, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [18], Training Batch Accuracy: [0.5543333604186773], Testing Batch Accuracy: [0.5341666914522648]
tensor(0.5849, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [19], Training Batch Accuracy: [0.5688333603739738], Testing Batch Accuracy: [0.52083335891366]
tensor(1.1399, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [20], Training Batch Accuracy: [0.5788333612680435], Testing Batch Accuracy: [0.5425000274553895]
tensor(0.7437, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [21], Training Batch Accuracy: [0.5973333611339331], Testing Batch Accuracy: [0.5616666946560145]
tensor(0.6529, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [22], Training Batch Accuracy: [0.6046666952222586], Testing Batch Accuracy: [0.5375000279396772]
tensor(0.9369, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [23], Training Batch Accuracy: [0.615166696831584], Testing Batch Accuracy: [0.5800000254064799]
tensor(0.9833, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [24], Training Batch Accuracy: [0.6258333615213633], Testing Batch Accuracy: [0.5883333593606949]
tensor(0.7231, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [25], Training Batch Accuracy: [0.6398333640396595], Testing Batch Accuracy: [0.5925000287592411]
tensor(1.5527, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [26], Training Batch Accuracy: [0.6485000302642584], Testing Batch Accuracy: [0.6116666935384274]
tensor(0.7703, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [27], Training Batch Accuracy: [0.6683333653956651], Testing Batch Accuracy: [0.6241666987538338]
tensor(1.0577, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [28], Training Batch Accuracy: [0.6718333654850721], Testing Batch Accuracy: [0.6208333641290664]
tensor(0.8757, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [29], Training Batch Accuracy: [0.6855000321567059], Testing Batch Accuracy: [0.6125000283122063]
tensor(0.6845, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [30], Training Batch Accuracy: [0.6938333684206008], Testing Batch Accuracy: [0.6491667002439498]
tensor(0.8463, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [31], Training Batch Accuracy: [0.7048333702236413], Testing Batch Accuracy: [0.6550000339746476]
tensor(0.6189, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [32], Training Batch Accuracy: [0.7110000364482403], Testing Batch Accuracy: [0.6708333622664213]
tensor(0.7093, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [33], Training Batch Accuracy: [0.719000036790967], Testing Batch Accuracy: [0.6375000271946192]
tensor(1.1077, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [34], Training Batch Accuracy: [0.7278333724290132], Testing Batch Accuracy: [0.6658333659172058]
tensor(0.5486, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [35], Training Batch Accuracy: [0.736000040024519], Testing Batch Accuracy: [0.6508333649486303]
tensor(1.4749, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [36], Training Batch Accuracy: [0.7393333733826876], Testing Batch Accuracy: [0.6716666996479035]
tensor(0.6030, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [37], Training Batch Accuracy: [0.7673333778977394], Testing Batch Accuracy: [0.6641666997224093]
tensor(0.8022, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [38], Training Batch Accuracy: [0.7623333763331175], Testing Batch Accuracy: [0.6333333648741245]
tensor(0.6607, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [39], Training Batch Accuracy: [0.7718333777040243], Testing Batch Accuracy: [0.6500000279396773]
tensor(0.5361, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [40], Training Batch Accuracy: [0.7768333797156811], Testing Batch Accuracy: [0.6925000324845314]
tensor(0.5917, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [41], Training Batch Accuracy: [0.7865000480413437], Testing Batch Accuracy: [0.6841666977852583]
tensor(0.7665, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [42], Training Batch Accuracy: [0.7938333785533905], Testing Batch Accuracy: [0.6908333659172058]
tensor(0.7780, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [43], Training Batch Accuracy: [0.8076667161285878], Testing Batch Accuracy: [0.7166667059063911]
tensor(0.9599, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [44], Training Batch Accuracy: [0.8091667157411575], Testing Batch Accuracy: [0.6808333672583103]
tensor(0.4649, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [45], Training Batch Accuracy: [0.8176667180657387], Testing Batch Accuracy: [0.6816666960716248]
tensor(0.4817, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [46], Training Batch Accuracy: [0.8250000505149364], Testing Batch Accuracy: [0.7150000326335431]
tensor(0.1736, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [47], Training Batch Accuracy: [0.8191667167842388], Testing Batch Accuracy: [0.6633333638310432]
tensor(0.5357, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [48], Training Batch Accuracy: [0.8396667203307152], Testing Batch Accuracy: [0.6983333684504032]
tensor(0.6665, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [49], Training Batch Accuracy: [0.8505000533163547], Testing Batch Accuracy: [0.6816667020320892]
tensor(0.6662, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [50], Training Batch Accuracy: [0.8500000508129597], Testing Batch Accuracy: [0.6766666974872351]
tensor(0.2696, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [51], Training Batch Accuracy: [0.8485000510513783], Testing Batch Accuracy: [0.7241667054593564]
tensor(0.1821, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [52], Training Batch Accuracy: [0.866500051766634], Testing Batch Accuracy: [0.730000039190054]
tensor(0.0564, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [53], Training Batch Accuracy: [0.8668333841860294], Testing Batch Accuracy: [0.706666698306799]
tensor(0.4486, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [54], Training Batch Accuracy: [0.8713333867490292], Testing Batch Accuracy: [0.7191667072474957]
tensor(0.2301, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [55], Training Batch Accuracy: [0.8818333849310875], Testing Batch Accuracy: [0.7175000347197056]
tensor(0.2494, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [56], Training Batch Accuracy: [0.8856667183339596], Testing Batch Accuracy: [0.716666703671217]
tensor(0.7229, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [57], Training Batch Accuracy: [0.8835000540316105], Testing Batch Accuracy: [0.7233333706855773]
tensor(0.2418, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [58], Training Batch Accuracy: [0.8920000480115413], Testing Batch Accuracy: [0.733333370834589]
tensor(0.3919, device='cuda:0', grad_fn=<NllLossBackward0>)
epoch: [59], Training Batch Accuracy: [0.8870000505447387], Testing Batch Accuracy: [0.6866667002439499]
tensor(0.7236, device='cuda:0', grad_fn=<NllLossBackward0>)

    """

# Regular expressions to extract the accuracy values
train_regex = r"Training Batch Accuracy: \[(.*?)\]"
test_regex = r"Testing Batch Accuracy: \[(.*?)\]"

# Lists to store the accuracy values
training_accuracy = []
testing_accuracy = []

# Extract training accuracy
find_train = re.findall(train_regex, accuracy_output)
for match in find_train:
    acc = float(match)
    training_accuracy.append(acc)

# Extract testing accuracy
find_test = re.findall(test_regex, accuracy_output)
for match in find_test:
    acc2 = float(match)
    testing_accuracy.append(acc2)

len(training_accuracy) == len(testing_accuracy)


# In[34]:


# Create a dictionary with the data
data = {
    'Training Accuracy': training_accuracy,
    'Testing Accuracy': testing_accuracy
}
df = pd.DataFrame(data)

# Plot accuracies
train_line = plt.plot(range(1,61), df['Training Accuracy'], label='Training Accuracy')
test_line = plt.plot(range(1,61), df['Testing Accuracy'], label='Testing Accuracy')
plt.grid(False)

# Add annotations
epochs_to_annotate = [5, 15, 25, 35, 45, 50, 55, 59]  #epochs to annotate
for epoch in epochs_to_annotate:
    plt.text(epoch, df['Training Accuracy'][epoch], f'{df["Training Accuracy"][epoch]:.2f}',
             ha='center', va='bottom', color=train_line[0].get_color(), fontsize=8)
    plt.text(epoch, df['Testing Accuracy'][epoch], f'{df["Testing Accuracy"][epoch]:.2f}',
             ha='center', va='top', color=test_line[0].get_color(), fontsize=8)

# Set plot title and labels
#plt.title('Training and Testing Accuracy')
plt.xlabel('Epochs', color='grey')
plt.ylabel('Accuracy', color='grey')

plt.legend()
plt.xticks(range(0,61, 5), color='grey')
plt.yticks(color='grey')

#Uncomment to View
#plt.show()
#Uncomment to save
plt.savefig("accuracy.png")


# In[35]:


#classification counts
confusion_matrix = np.array([[140., 7., 1., 17., 18., 17.],
          [12., 107., 3., 5., 55., 18.],
          [8., 12., 116., 27., 29., 8.],
          [4., 4., 2., 169., 10., 11.],
          [20., 9., 2., 21., 137., 11.],
          [11., 7., 1., 13., 13., 155.]], dtype=int)

# Define class labels
languages = ["de", "en", "es", "fr", "nl", "pt"]

# Create matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap="Blues", vmin=0, vmax=30, xticklabels=languages, yticklabels=languages)

# Set plot title and labels
#plt.label("confusion matrix")
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')

#Uncomment to View
#plt.show()
#Uncomment to save
plt.savefig("error-analysis.png")


# In[36]:


# Calculate precision, recall, and F1 score
precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
f1_score = 2 * (precision * recall) / (precision + recall)

# Create a DataFrame to display the results
metrics_table = pd.DataFrame({'Precision': precision, 'Recall': recall, 'F1 Score': f1_score}, index=languages)
metrics_table = metrics_table.round(2)
print(metrics_table)

# Convert metrics table to Overleaf table format
table = tabulate(metrics_table, headers='keys', tablefmt='latex')

# Save the table as a LaTeX file
with open('metrics_table.tex', 'w') as f:
    f.write(table)


# In[37]:


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


# In[38]:


batch_size = 15
model_graph = draw_graph(model, input_size=(batch_size, 1, 40000),
                         device='meta', expand_nested=True,
                         hide_module_functions=False,
                         save_graph=True, filename='architecture_full')
model_graph.visual_graph


# In[39]:


class LanguageClassifier_cropped(nn.Module):
    def __init__(self):

        super(LanguageClassifier_cropped, self).__init__()

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
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.maxpool3(x)

        x = self.conv4(x)
        x = self.maxpool4(x)

        x = self.adaptive_pool(x)
        x = self.flatten(x)


        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)

        return x



model_cropped = LanguageClassifier_cropped()


# In[40]:


batch_size = 15
model_graph = draw_graph(model_cropped, input_size=(batch_size, 1, 40000),
                         device='meta', expand_nested=True,
                         hide_module_functions=False,
                         save_graph=True, filename='architecture_cropped')
model_graph.visual_graph

