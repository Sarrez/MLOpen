from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torchvision.transforms.functional as functional
import torch.nn.functional as fn
import torchvision.transforms as transforms
import torch.optim as optim # optimzer
import os
import os.path
import torchvision
from mlopenapp.utils import io_handler as io
import zipfile
from mlopenapp.pipelines.NeuralNetworkFeedForward.alexnet import AlexNet
import pandas as pd
from mlopenapp.utils import plotter
import torch
import torch.nn as nn
import torch.nn.functional as fn
import shutil


def unzip_imgs(input, path_to_ds, unzip_path):
    path_to_dataset = path_to_ds + str(input)
    if not(os.path.exists(unzip_path)):
        os.makedirs(unzip_path)
    print('Extracting images at ', unzip_path)
    newpath=unzip_path+str(input).removesuffix('.zip')
    if not(os.path.exists(newpath)):
        with zipfile.ZipFile(path_to_dataset,"r") as zip_ref:
            zip_ref.extractall(unzip_path)
    
    return newpath
        
    
def load_data(train_path, img_size, batch_size):
    
    ''' Loads datasets for training as DataLoaders '''
    
    #unzip inpt if it's not unzipped    
    path = unzip_imgs(train_path, 'mlopenapp/data/user_data/', 'mlopenapp/data/user_data/unzipped-imgs/')
    #Transforms
    datasetTransforms = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
        transforms.RandomHorizontalFlip(p=0.5)
    ])
    #Image paths
    #Load train and validation sets
    traindata=torchvision.datasets.ImageFolder(root=path+'/train', transform=datasetTransforms)
    classes = traindata.classes
    train_size = int(len(traindata)* 0.8)
    valid_size = len(traindata) - train_size
    validation, train = torch.utils.data.random_split(traindata, [valid_size, train_size], generator=torch.Generator().manual_seed(42))
    loadvaldata = torch.utils.data.DataLoader(validation, batch_size=batch_size, shuffle=True,num_workers=1)
    loadtraindata = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True,num_workers=1)
        
    return loadtraindata,loadvaldata,classes

def train_model(model, train_dl, valid_dl, optimizer, criterion, device='cpu'):
    
    ''' Implements the training of a model for an epoch '''
    loss_ep = 0
    valid_ep = 0
    epoch_loss = 0
    validation_loss = 0
    
    model.train()
    for batch_idx, (data, targets) in enumerate(train_dl):
        data = data.to(device=device)
        targets = targets.to(device=device)
        #Forward Pass
        optimizer.zero_grad()
        scores, probs = model(data)
        loss = criterion(scores,targets)
        loss.backward()
        optimizer.step()
        loss_ep += loss.item()
        epoch_loss = loss_ep/len(train_dl)
        
    model.eval()
    for batch_idx, (data, targets) in enumerate(valid_dl):
        data = data.to(device=device)
        targets = targets.to(device=device)
        scores, probs = model(data)
        loss = criterion(scores,targets)
        valid_ep += loss.item()
        validation_loss = valid_ep/len(valid_dl)
    return model, epoch_loss, optimizer

def train_loop(train_path, img_size, batch_size, criterion, optimizer, num_epochs):
    
    ''' Implements the training of a model for a given number of epochs '''
    
    batch_size = 64
    img_size = 227
    loadtraindata,loadvaldata,classes = load_data(train_path, img_size, batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AlexNet() 
    model = model.to(device='cpu') 
    learning_rate = 1e-4 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr= learning_rate) 
    losses = []
    print('Starting training for ', num_epochs,'epochs')
    for epoch in range(num_epochs):
        model, epoch_loss, optimizer = train_model(model, loadtraindata, loadvaldata, optimizer,criterion)
        print(f"Loss in epoch {epoch} :::: {epoch_loss}")
        losses.append(epoch_loss)

    return model, optimizer, losses, classes

def load_run_dataset(path):
    
    ''' Loads the dataset used to run a trained model '''
    unzip_path = 'mlopenapp/data/user_data/unzipped-imgs/'+ str(path).removesuffix('.zip') + '/'
    unzipped_path = unzip_imgs(path,'mlopenapp/data/user_data/',unzip_path)
    dst_path = 'mlopenapp/static/images/'+ path.removesuffix('.zip')
    if not (os.path.exists(dst_path)):
        shutil.copytree(unzip_path, dst_path)
    
    testTransforms = transforms.Compose([
        transforms.Resize((227,227)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ])
    ds = torchvision.datasets.ImageFolder(dst_path, transform=testTransforms)
    loadrundata = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False,num_workers=1)
    return ds, loadrundata

def predict_images(dataset, model, classlist):
    
    ''' Function to predict images using a trained model '''
    #set up results storage
    classes = []
    #make predictions
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(dataset):
            outputs, probs=model(data)
            index = probs.data.cpu().numpy().argmax()
            classes.append(classlist[index])

    return classes

def get_df(filenames, classes):
    
    '''Forms prediction results into a pandas DataFrame'''

    results = {}
    results['imagename'] = filenames
    results['class'] = classes
    df = pd.DataFrame(results)
    return df

def train(inpt, params=None):
    
    '''Function to train the model in the platform'''
    batch_size = 64
    img_size = 227
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AlexNet() 
    model = model.to(device='cpu') 
    learning_rate = 1e-4 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr= learning_rate) 
    
    model, optimizer, losses, classes = train_loop(inpt.__str__(), img_size, batch_size, criterion, optimizer, num_epochs = 1)
    #save model
    models = [(model, 'img-classification-model')]
    args = [(classes, 'img-classification-classlist')]
    io.save_pipeline(models, args, os.path.basename(__file__))
    return model

def run_pipeline(input, model, args, params=None):
    
    '''Function to run the model in the platform'''
    preds={'graphs':[], 'imgs':[]}
    #prepare input     
    dataset, dataloader = load_run_dataset(str(input))

    #predict images
    classlist = args['img-classification-classlist']
    classes = predict_images(dataloader, model, classlist)
    
    #prepare results format
    filenames = []
    for filename,_ in dataset.imgs:
        filenames.append(filename.replace('mlopenapp',''))

    #format results
    df = get_df (filenames, classes)
    counts = df['class'].value_counts().sort_index().sort_index()
    unique_classes = df['class'].unique().tolist()
    res = {'Name':[], 'Images':[], 'Count':[]}
    counts = df['class'].value_counts().sort_index().sort_index().tolist()
    unique_classes = sorted(df['class'].unique().tolist())
    num_classes = len(unique_classes)
    groups = df.groupby('class')
    finalDf = pd.DataFrame()
    for i in range (0,  num_classes):
        class_index = unique_classes[i]
        finalDf = groups.get_group(class_index)
        count = finalDf['class'].value_counts().tolist()
        finalDf = finalDf.drop(['class'], axis=1)
        res['Name'].append(class_index)
        res['Images'].append(finalDf['imagename'].tolist())
        res['Count'].append(counts[i])

    imgs_dict = {}
    for i in range(len(res['Name'])):
        imgs_dict.update({res['Name'][i]:res['Images'][i]})

    preds['imgs'].append(imgs_dict)
    preds['graphs'] += plotter.bar(res,'Name','Count')
    return preds