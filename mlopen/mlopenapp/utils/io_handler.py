import sys
import os
import pickle
import datetime
from re import search

from mlopen.settings import BASE_DIR
from ..models import models
from django.core.files import File
from .. import constants
from ..models import storages
from django.conf import settings

from django.core.files.storage import FileSystemStorage


def save(arg_object, name, save_to_db=False, type=None):
    try:            
        output = open(constants.FILE_DIRS[type] + '/' + name + '.pkl', 'wb')
        #print('Will be saving ', output)
        pickle.dump(arg_object, output, pickle.HIGHEST_PROTOCOL)
        output.close()
        if save_to_db:
            output = open(constants.FILE_DIRS[type] + '/' + name + '.pkl', 'rb')
            #print('output is: ', output)
            if(type=='graphs'):
                filefield, _ = constants.FILE_TYPES[type].objects.get_or_create(
                    name=name,
                    defaults={
                        'updated_at': datetime.datetime.now(),
                        'graphs': File(output)
                    }
                )
            else:
                filefield, _ = constants.FILE_TYPES[type].objects.get_or_create(
                    name=name,
                    defaults={
                        'created_at': datetime.datetime.now(),
                        'updated_at': datetime.datetime.now(),
                        'file': File(output)
                    }
                )
            filefield.save()
            output.close()
            return filefield
        return True
    except Exception as e:
        print(f"exception is:", e)
        return False


def load(name, type):
    try:
        with open(os.path.join(constants.FILE_DIRS[type], name), 'rb') as input:
            ret = pickle.load(input)
        return ret
    except Exception as e:
        print(e)
        return False

def save_graphs(graphs, name):
    temp = save(graphs, name, True, 'graphs')
    
    
def save_pipeline(models, args, name):
    pip_models = []
    pip_args = []
    if not(os.path.exists('mlopenapp/storage/args')):
        os.makedirs('mlopenapp/storage/args')
    if not(os.path.exists('mlopenapp/storage/models')):
        os.makedirs('mlopenapp/storage/models')
    for model in models:
        temp = save(model[0], model[1], True, 'model')
        print(f'Temp model: ', temp)
        if type(temp) == bool:
            return False
        pip_models.append(temp)
    #print(len(args))
    for arg in args:
        temp = save(arg[0], arg[1], True, 'arg')
        if type(temp) == bool:
            return False
        pip_args.append(temp)
    name = name[:-3] if name.endswith(".py") else name
    print(f"name is: ",name)
    pipeline, _ = constants.FILE_TYPES['pipeline'].objects.get_or_create(
        control=name,
        defaults={
            'name': name,
            'created_at': datetime.datetime.now(),
            'updated_at': datetime.datetime.now()
        }
    )
    pipeline.save()
    for model in pip_models:
        pipeline.ml_models.add(model)
        print('pip model: ', model)
    for arg in pip_args:
        pipeline.ml_args.add(arg)
    pipeline.save()
    print('saved')


def get_pipeline_list():
    pipeline_list = []
    print(pipeline_list)
    for filename in os.listdir(constants.CONTROL_DIR):
        if filename.endswith("control.py"):
            pipeline_list.append(filename[:-3])
    return pipeline_list


def save_pipeline_file(f):
    with open(os.path.join(constants.CONTROL_DIR, f.name), 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)


def load_pipeline(pipeline):
    try:
        model_qs = pipeline.get_models()
        #model_qs.delete()
        print('model qs', model_qs)
        args_qs = pipeline.get_args()
        model = None
        args = {}
        for m in model_qs:
            if search('torch', m.name):
                model = m.file.path
            else:
            #print('name of model file',m.file.path)
                model = pickle.load(m.file.open('rb'))
            #print("Loaded model ", model)
        for ar in args_qs:
            args[ar.name] = pickle.load(ar.file.open('rb'))
        return model, args
    except Exception as e:
        print(e)
        return False


def save_pipeline_files(pipeline, files):
    if not files:
        return True
    if pipeline.endswith(".py"):
        pipeline = pipeline[:-3]
    dir_name = os.path.join(constants.CONTROL_DIR, pipeline)
    try:
        # Create target Directory
        os.mkdir(dir_name)
    except FileExistsError:
        return False, "Directory already exists."
    for f in files:
        with open(os.path.join(dir_name, f.name), 'wb+') as destination:
            for chunk in f.chunks():
                destination.write(chunk)
    return True, "Directory and files successfully created."
