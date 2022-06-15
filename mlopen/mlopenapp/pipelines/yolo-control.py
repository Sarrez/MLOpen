from cmath import cos
from gc import callbacks
import zipfile
from PIL import Image
from sqlalchemy import null
import torch
import numpy as np
from mlopenapp.utils import plotter
import pandas as pd
import os
from mlopenapp.utils import io_handler as io
pretrained=1
from mlopenapp.yolov5 import train as yolo_train
from mlopenapp.yolov5.utils.callbacks import Callbacks 
import glob
import os
#reads images from a zip file
#'input' is the path to the zip file
#returns list of images and list of image names
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
    
    ''' Reads images from a zip file and loads them in a list
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
    unzipped = "mlopen/mlopenapp/data/user_data/unzipped-imgs/" 
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
    savepath = path
    imgnames = [('image'+str(i)+'.jpg') for i in range (len(os.listdir(savepath)))]
    return imgnames
#prepares images for heatmap per class
#Constructs heatmaps 
#Constructs heatmaps 
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
    #dt = results.pandas()
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
def train(input, args):
    #unzip images
    #unzip_imgs(input)
    #name = str(input).removesuffix('.zip')
    #path = "mlopen/mlopenapp/data/user_data/unzipped-imgs/"+name
    #print(path)

    #train model
    #os.system('git clone https://github.com/ultralytics/yolov5')
    name = 'yolo_results_390'
    path = ' mlopen/mlopenapp/yolov5/runs/train/'+name+'/best.pt'
    #os.system('pip install -U -r mlopen/mlopenapp/yolov5/requirements.txt')
    #os.system('ls mlopen/mlopenapp/data/user_data/unzipped-imgs/imgnet2012sample/images/')
    os.system('pip install sqlalchemy')
    hyp = 'mlopen/mlopenapp/pipelines/yolo-control/hyp.yaml'
    save_path = 'mlopen/mlopenapp/yolov5/runs/train/yolo_results' 
    opt = Options(save_path, 'mlopen/mlopenapp/yolov5s.pt',
     'mlopen/mlopenapp/pipelines/yolo-control/imgnet2012sample.yaml', 
      epochs=1, batch_size=32, imgsz=150, workers=1, name='results',
      single_cls=False, evolve= null, cfg='yolov5/models/yolov5s.yaml', 
      resume=False, noval=False, nosave=False, freeze= [0],pretrained=False, optimizer='SDG',
      cos_lr=False, rect=False, noautoanchor=False, sync_bn=False, cache='ram',image_weights=False,project = 'mlopen/mlopenapp/yolov5/runs/train')
    callbacks = Callbacks()
    print('training yolo')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    yolo_train.train( hyp, opt, device, callbacks)
    print('trained yolo')
    models = []
    args = [('mlopen/mlopenapp/yolov5/runs/train/yolo_results_34/best.pt','best-weights.pkl')]
    path_to_weights = 'mlopen/mlopenapp/yolov5/runs/train/yolo_results_34/best.pt'
    io.save_pipeline(models, args, os.path.basename(__file__))
    return path_to_weights

def run_pipeline(input, model, args, params=None):
    import pickle
    #path = 'M:/MLOpen2/mlopen/mlopenapp/yolov5/runs/train/'+name+'/weights/best.pt'
    os.system('pip install -U -r mlopen/mlopenapp/yolov5/requirements.txt')
    #if(pretrained):
    #    print('pretrained')
    #model = yolo_detect.run('ultralytics/yolov5','yolov5s', classes=80)
    print('Model is ', model)
    #print('args is ', (args))
    #if(bool(args)):
        #using custom weights from training
        #print('Using custom weights from training')
        #weights_path = args['best-weights.pkl']
        #model = torch.hub.load('ultralytics/yolov5', weights_path , classes=80)
        #model = torch.hub.load('ultralytics/yolov5', 'custom', weights_path)
    #else:
        #pretrained using yolov5s weights
    print("running model")
    model = torch.hub.load('ultralytics/yolov5','yolov5s', force_reload=True)
    #imgnames = get_image_names('runs/detect/exp27')
    imgs = load_images(input)
    preds = {'graphs': [], 'text':''}
    #model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True) 
    print('')
    results = model(imgs)
    results.save()
    #get results as dataframes
    #path to original results
    filelist = glob.glob('runs/detect/*') # * means all if need specific format then *.csv
    latest_file = max(filelist, key=os.path.getctime)
    #imgnames = get_image_names('detect/exp40/')
    print(latest_file)
    imgnames = get_image_names(latest_file+'/')
    frames = get_results(results,len(imgs),imgnames)
    graph_data = collect_imgs(frames)

    #extract data from dataframes
    #classes = graph_data['Name']
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
    #print(graph_data['Name'])
    imgs_dict = plotter.populate_images(latest_file+'/', graph_data['Images'], graph_data['Name'])
    #print(img_list)
    preds.update({'imgs': [imgs_dict]})
    #preds['imgs'] += plotter.show_images("detect/expjpeg", graph_data['Images'][0], graph_data['Name'][0][0])
    print('the size of img_list is',len(imgs_dict))
    #print(img_dict['person'])
    return preds