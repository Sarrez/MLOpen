import zipfile
from PIL import Image
from sqlalchemy import null
import torch
import numpy as np
from mlopenapp.utils import plotter
import pandas as pd
import os
from mlopenapp.utils import io_handler as io
os.system('git clone https://github.com/ultralytics/yolov5')
#os.system('pip install -U -r yolov5/requirements.txt')
from yolov5 import train as yolo_train
from yolov5.utils.callbacks import Callbacks 
import glob
import os

class Options:
    def __init__(self, save_dir, weights, data, epochs, batch_size, imgsz, workers, name, single_cls,
    evolve, cfg, resume, noval, nosave, freeze, pretrained, optimizer,cos_lr,rect,noautoanchor,sync_bn,
    cache,image_weights,project, save_period = -1,local_rank= -1,entity= null,upload_dataset=False,bbox_interval=-1,
    artifact_alias='latest',patience=100,label_smoothing=100,quad=False, multi_scale = False,
    exist_ok= False, bucket=''):
        self.save_period = save_period
        self.local_rank = local_rank
        self.entity = entity
        self.upload_dataset = upload_dataset
        self.bbox_interval = bbox_interval
        self.artifact_alias = artifact_alias
        self.patience = patience
        self.label_smoothing = label_smoothing
        self.quad = quad
        self.multi_scale = multi_scale
        self.exist_ok = exist_ok
        self.bucket = bucket
        self.project = project  
        self.save_dir = save_dir
        self.weights = weights
        self.data = data
        self.epochs = epochs
        self.batch_size = batch_size
        self.imgsz = imgsz
        self.workers = workers
        self.name = name
        self.single_cls = single_cls
        self.evolve = evolve
        self.cfg = cfg
        self.resume = resume
        self.noval = noval
        self.nosave = nosave
        self.freeze = freeze
        self.pretrained = pretrained
        self.optimizer = optimizer
        self.cos_lr = cos_lr
        self.rect = rect
        self.noautoanchor = noautoanchor
        self.sync_bn = sync_bn
        self.cache = cache
        self.image_weights = image_weights

def load_images(input):
    
    ''' Reads images from a zip file and loads them in a list used to run the YOLO pipeline
        :param input: Path to zip file.   '''
        
    print("To input einai", input)
    print('loading images..')
    zip = zipfile.ZipFile(input)
    list = zip.infolist()
    print('loaded images..')
    imgs=[]
    
    for f in list:
        file = zip.open(f)
        img = Image.open(file)
        img = img.resize((150,150))
        imgs += [img]
    print('returning images..')
    return imgs
def unzip_imgs(input):
    if not(os.path.exists('mlopenapp/data/user_data/unzipped-imgs/')):
        os.makedirs('mlopenapp/data/user_data/unzipped-imgs/')
    unzipped = "mlopenapp/data/user_data/unzipped-imgs/" 
    print('Extracting images at ', unzipped)
    with zipfile.ZipFile(input,"r") as zip_ref:
        zip_ref.extractall(unzipped)

def detect_bbox(img,xmin,ymin,xmax,ymax):
    
    ''' Stacks bounding boxes of the a class on an initially empty image.
        :param img: 2D numpy array '''
    
    for i in range(img.shape[0]):
        height=i
        for j in range(img.shape[1]):
            width=j
            if(width<xmin or width>xmax):
                img[i][j]+=1
            elif(height<ymin or height>ymax):
                img[i][j]+=1
    return img

def get_image_names(path):
    staticpath = path.replace('mlopenapp','')
    print('to static path einai', staticpath)
    imgnames = [(staticpath+'image'+str(i)+'.jpg') for i in range (len(os.listdir(path)))]
    return imgnames

def collect_imgs(frames):
    #classlist contains imglists
    imglist=[]
    total=0
    total_confidence=0
    heatmaps = {"Name": [],"Heatmap":[],"Average Confidence":[], "Images":[]}
    # for each frame
    for i in range (0, len(frames)):   
        #initialize empty image as numpy array 
        final_img = np.zeros((150,150))
        imglist=[]
        frame = frames[i]
        #for each row in frame
        for i, row in frame.iterrows():
            final_img = detect_bbox(final_img,row.xmin,row.ymin,row.xmax,row.ymax)
            total_confidence += row.confidence
            if row.imgname not in imglist:
                imglist.append(row.imgname) 
        #edit results
        avg_confidence=0
        final_img = final_img/len(frame)
        avg_confidence = total_confidence/len(frame)
        #append results to dictionary
        heatmaps["Name"].append(frame['name'].unique())
        heatmaps["Heatmap"].append(final_img)
        heatmaps["Average Confidence"].append(avg_confidence)
        heatmaps["Images"].append(imglist)
        
    return heatmaps

def get_results(results,num_of_imgs,image_names):
    df = results.pandas().xyxy[0]
    df['imgname'] = image_names[0]
    merged_results = pd.DataFrame()
    merged_results = df
    
    #for each dataframe of results add respective image name and concat 
    for i in range (1,num_of_imgs):
        if(len(results.pandas().xyxy[i])>0):
            df = results.pandas().xyxy[i]
            df['imgname'] = image_names[i]
            merged_results = pd.concat([merged_results,df])

    merged_results = merged_results.sort_values(by=['class'])
    
    split = []
    unique_classes = merged_results['class'].unique()
    num_classes = len(unique_classes)
    groups = merged_results.groupby('class')
    for i in range (0,  num_classes):
        class_index = unique_classes[i]
        df = groups.get_group(class_index)
        split += [df]
        
    return split
def train(input, args=[]):
    #unzip images
    datapath = 'mlopenapp/data/user_data/'+str(input)
    print('to path tou dataset einai', datapath)
    unzip_imgs(datapath)
    #train model
    hyp = 'mlopenapp/pipelines/yolo-control/hyp.yaml'
    save_path = 'mlopenapp/pipelines/yolo_results/results' 
    opt = Options(save_path, 'mlopenapp/pipelines/yolo-control/yolov5s.pt',
     'mlopenapp/pipelines/yolo-control/imgnet2012sample.yaml', 
      epochs=1, batch_size=32, imgsz=150, workers=1, name='results',
      single_cls=False, evolve= null, cfg='yolov5/models/yolov5s.yaml', 
      resume=False, noval=False, nosave=False, freeze= [0],pretrained=False, optimizer='SDG',
      cos_lr=False, rect=False, noautoanchor=False, sync_bn=False, cache='ram',image_weights=False,project = 'mlopenapp/pipelines/yolo_results/')
    callbacks = Callbacks()
    print('training yolo')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    yolo_train.train( hyp, opt, device, callbacks)
    print('trained yolo')
    models = []
    path_to_weights = 'mlopenapp/pipelines/yolo_results/results/weights/best.pt'
    args = [(path_to_weights,'best-weights')]
    io.save_pipeline(models, args, os.path.basename(__file__))
    return path_to_weights

def run_pipeline(input, model, args, params=None):
    
    preds = {'graphs': [], 'text':'','imgs':[]}
    if(bool(args)):
        #using custom weights from training
        print('Using custom weights from training')
        weights_path = args['best-weights']
        model = torch.hub.load('ultralytics/yolov5', 'custom', weights_path)
    else:
        #pretrained using yolov5s weights
        model = torch.hub.load('ultralytics/yolov5','yolov5s', force_reload=True)
        
    imgs = load_images(input)
    results = model(imgs)
    results.save(save_dir='mlopenapp/static/images/runs/detect/exp')
    
    #get results as dataframes
    filelist = glob.glob('mlopenapp/static/images/runs/detect/*') 
    latest_file = max(filelist, key=os.path.getctime)
    imgnames = get_image_names(latest_file+'/')
    frames = get_results(results,len(imgs),imgnames)
    graph_data = collect_imgs(frames)

    #extract data from dataframes
    print("making heatmaps")
    classes = []
    for i in range(0, len(graph_data['Name'])):
        classes.append(graph_data['Name'][i][0])
    print('Classes: ', classes)
    conf_list = graph_data['Average Confidence']
    heatmaps = graph_data['Heatmap']
    data = {'Class':classes,'Average confidence':conf_list}
    df = pd.DataFrame(data)
    
    #add graphs
    print("adding graphs")
    preds['graphs'] += plotter.bar(df,df.columns[0],df.columns[1])
    preds['graphs'] += plotter.custom_heatmap(heatmaps, classes)
    imgs_dict = {}
    for i in range(len(classes)):
        imglist = graph_data['Images'][i]
        print(imglist)
        imgs_dict.update({classes[i]:imglist})
    preds['imgs'].append(imgs_dict)
    return preds