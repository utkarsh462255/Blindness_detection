import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import seaborn as sns

#from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import cv2 # Computer vision module for image processing
import os  # os module with methods for interacting with the operating system

import timm
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader,random_split
from torchvision import datasets, transforms

from PIL import Image
import torchvision
from tqdm.notebook import tqdm

seed = 42
torch.manual_seed(seed)
print(torch.__version__)
device = "cuda" if torch.cuda.is_available() else "cpu"
print('Active Device:',device)

train_dir = "/kaggle/input/aptos2019-blindness-detection/train_images"
test_dir = "/kaggle/input/aptos2019-blindness-detection/test_images"
print("Training Folder path:  ",train_dir)
print("Testing Folder path:   ",test_dir)

train = pd.read_csv('/kaggle/input/aptos2019-blindness-detection/train.csv')
test  = pd.read_csv('/kaggle/input/aptos2019-blindness-detection/test.csv')
print(train.head())
print(test.head())

train.diagnosis.value_counts()
sns.countplot(x=train['diagnosis'])
train, val = train_test_split(train, test_size=0.25, stratify=train.diagnosis, random_state=seed)
train.diagnosis.value_counts()
val.diagnosis.value_counts()

t_dir = "/kaggle/input/aptos2019-blindness-detection/train_images/"
plt.figure(figsize=[15,15])
i = 1
for img_name in train['id_code'][:10]:
    img = mpimg.imread(t_dir + img_name + '.png')
    plt.subplot(6,5,i)
    plt.imshow(img)
    i += 1
plt.show()

class Customized_Data(Dataset):
    def __init__(self, image_folder, csv_file, transform=None):
        super().__init__()
        self.image_folder = image_folder
        self.label_csv = csv_file
        self.transform = transform
    
    def __getitem__(self, idx):
        img_name = self.label_csv.id_code.values[idx] + '.png'
        img_path = os.path.join(self.image_folder, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        
        cropped_image = self.transform(image=image)["image"]
        label = self.label_csv.diagnosis.values[idx]
        
        return cropped_image, label
        

    def __len__(self):
        return len(self.label_csv)
    
    import albumentations as A
from albumentations.pytorch import ToTensorV2

# Define augmentation transformations
data_transforms = A.Compose([
    A.Resize(height=512, width=512),
    #A.RandomCrop(width=512, height=512),
    A.RandomBrightnessContrast(p=0.2),
    #A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=1.0),
    A.Blur(p=1.0),
    #A.Rotate(limit=180, p=1.0),
    #A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=0, p=1.0),
    #A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=0, p=1.0),
    #A.HorizontalFlip(p=1.0),
    ToTensorV2(),
])
def show_images(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    batch = next(iter(loader))
    images, labels = batch
    grid = torchvision.utils.make_grid(images, nrow=4)
    plt.figure(figsize=(10,10))
    plt.imshow(np.transpose(grid, (1,2,0)))
    print('labels : ', labels)
    
show_images(custom_train)

train_dataloader = DataLoader(dataset=custom_train,batch_size=16, shuffle=True) 
val_dataloader   = DataLoader(dataset=custom_val,  batch_size=16, shuffle=True)

image, label = next(iter(train_dataloader))
print(f"Image shape: {image.shape} -> [batch_size, color_channels, height, width]")
print(f"Label shape: {label.shape}")

image, label = next(iter(val_dataloader))
print(f"Image shape: {image.shape} -> [batch_size, color_channels, height, width]")
print(f"Label shape: {label.shape}")

model1 = timm.list_models('*tf_efficientnet_lite4*')
print(model1)

efficientnet = timm.create_model('tf_efficientnet_lite4', pretrained = True, num_classes=5)

# Initializing Error and Optimizer

# Cross Entropy Loss 
error = nn.CrossEntropyLoss()

# AdamW Optimizer
learning_rate = 0.001
optimizer = torch.optim.Adam(efficientnet.parameters(),lr=learning_rate)

model = efficientnet.to(device)
print(model.to(device))
device

from torchinfo import summary
summary(model)

if torch.cuda.is_available():
    device = torch.device("cuda") 
else:
    device = torch.device("cpu")
print(device)

def Cmatrix(rater_a, rater_b, min_rating=None, max_rating=None):
    #assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat

def histogram(ratings, min_rating=None, max_rating=None):
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings

def quadratic_weighted_kappa(y, y_pred):
    rater_a = y
    rater_b = y_pred
    min_rating=None
    max_rating=None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    #assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
        conf_mat = Cmatrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator / denominator)

epochs = 10
output1 = []
label1  = []

# Training Loop
for epoch in tqdm(range(epochs)):
    
    epoch_loss = 0
    epoch_accuracy = 0
    best_loss = float('inf')
    for image, label in train_dataloader:
        model.train()
        
        image  = image.to(device)
        label  = label.to(device)
        
        output = model(image.float())
        loss = error(output, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        acc = ((output.argmax(dim=1) == label).float().mean())
        epoch_accuracy += acc/len(train_dataloader)
        epoch_loss += loss/len(train_dataloader)
        preds = output.argmax(dim=1)
        output1.extend(preds.view(-1).detach().to("cpu").numpy())
        label1.extend(label.view(-1).detach().to("cpu").numpy())
        #print(output1)
        #print(label1)
        #print(quadratic_weighted_kappa(label1,output1))
    
    qwk = quadratic_weighted_kappa(label1,output1)
    print('Epoch : {}, train accuracy : {}, train loss : {}, kappa_value : {}'.format(epoch+1,epoch_accuracy,epoch_loss,qwk))


total_correct = 0
total_samples = 0
with torch.no_grad():
    for image, label in val_dataloader:
        model.eval()
        
        image  = image.to(device)
        label  = label.to(device)
        
        outputs = model(image.float())
        val_loss = error(outputs, label)
        
        predicted = torch.argmax(outputs, 1)
        total_samples += label.size(0)
        total_correct += (predicted == label).sum().item()
        

    # Check for overfitting
    if val_loss < best_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model_effnetb0.pth')
    
validation_accuracy = total_correct / total_samples

print("Training Accuracy  : {:.2f}%" .format(epoch_accuracy * 100))
print("Validation Accuracy: {:.2f}%" .format(validation_accuracy * 100))

import matplotlib.pyplot as plt
models = ['ResNet-18', 'Tf_efficientnet_b4', 'Tf_efficientnet_lite4', 'Efficientnet_b0','Inception-v4-1', 
          'Inception-v4-2', 'Seresnext50_32x4d-1', 'Seresnext50_32x4d-2', 'Seresnext50_32x4d-3','Ensemble1','Ensemble2','Ensemble3','Ensemble4','Tf_efficientnet_lite4-Aug']

train_acc     = [0.7933, 0.9171, 0.9362, 0.9493, 0.9113, 0.8474, 0.9132, 0.8546, 0.9322,0.0,0.0,0.0,0.0,0.9007]
val_acc       = [0.7686, 0.8136, 0.8275, 0.8221, 0.8349, 0.7915, 0.8445, 0.7981, 0.8242,0.8302,0.0,0.0,0.0,0.8231]
LB_priv_score = [0.7233, 0.8397, 0.8461, 0.8417, 0.8039, 0.7954, 0.8412, 0.7690, 0.8332,0.8780,0.8600,0.8271,0.8677,0.8467]
LB_pub_score  = [0.4892, 0.6648, 0.6322, 0.6420, 0.5794, 0.5683, 0.5332, 0.5974, 0.6133,0.6658,0.6559,0.5869,0.6599,0.6435]

x = range(len(models))

fig, ax = plt.subplots(figsize=(10, 12))
bar_width=0.2
ax.barh(x, train_acc, height=0.2, label='Train Acc', color='b')
ax.barh([pos + bar_width for pos in x], val_acc, height=0.2, label='Val Acc', color='g')
ax.barh([pos + 2 * bar_width for pos in x], LB_priv_score, height=0.2, label='Private score', color='r')
ax.barh([pos + 3 * bar_width for pos in x], LB_pub_score, height=0.2, label='Public score', color='y')

def add_values(rects, ax):
    for rect in rects:
        width = rect.get_width()
        ax.annotate(f'{width:.2%}',  # Display as percentage with no decimal places
                    xy=(width, rect.get_y() + rect.get_height() / 2),
                    xytext=(5, 0),  # 5 points horizontal offset
                    textcoords='offset points',
                    va='center')

add_values(ax.containers[0], ax)
add_values(ax.containers[1], ax)
add_values(ax.containers[2], ax)
add_values(ax.containers[3], ax)

# Set labels and title
ax.set_ylabel('Models')
ax.set_xlabel('Scores')
ax.set_title('Model Performance Comparison')
ax.set_yticks([pos + 1.5 * bar_width for pos in x])
ax.set_yticklabels(models)  # Keep y-axis labels vertical
ax.invert_yaxis()  # Invert y-axis to have the highest score at the top
ax.legend(loc='upper right', bbox_to_anchor=(1, 1))  # Move legend to the top right corner

# Customize x-axis ticks
ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
ax.set_xticklabels(['0', '10', '20' ,'30', '40', '50', '60', '70', '80', '90', '100','110'])

# Add a little space between each model's bars
ax.set_xlim(0, 1.2)

# Show the plot
plt.tight_layout()
plt.show()

import pandas as pd
# Comparison of Cross Validation Scores and Leaderboard Score
Comparison_Table = {'Model Name '     : ['ResNet-18', 'Tf_efficientnet_b4', 'Tf_efficientnet_lite4', 'Efficientnet_b0','Inception-v4-1', 'Inception-v4-2', 'Seresnext50_32x4d-1', 'Seresnext50_32x4d-2', 'Seresnext50_32x4d-3','Ensemble1','Ensemble2','Ensemble3','Ensemble4','Tf_efficientnet_lite4-Aug'],
                    'Train Accuracy'  : [0.7933, 0.9171, 0.9299, 0.9493, 0.9113, 0.8474, 0.9132, 0.8546, 0.9322,'--','--','--','--',0.9007],
                    'Validation Accuracy'    : [0.7686, 0.8136, 0.8242, 0.8221, 0.8349, 0.7915, 0.8445, 0.7981, 0.8242,'--','--','--','--',0.8231],
                    'LB Private Score'       : [0.7233, 0.8397, 0.8461, 0.8417, 0.8039, 0.7954, 0.8412, 0.7690, 0.8332,0.8780,0.8600,0.8271,0.8677,0.8467],
                    'LB Public Score'        : [0.4892, 0.6648, 0.6322, 0.6420, 0.5794, 0.5683, 0.5332, 0.5974, 0.6133,0.6658,0.6559,0.5869,0.6599,0.6435],}
 
# Create a DataFrame from the dictionary
df = pd.DataFrame(Comparison_Table)

# Print the DataFrame
df = df.sort_values(by=['LB Private Score'],ascending=False)
df = df.reset_index(drop=True)
df.index += 1
df


