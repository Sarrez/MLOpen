from mlopenapp.models.models import MLPipeline as Pipeline
import subprocess
from mlopenapp.models.models import MLPipeline as Pipeline
from mlopen.celery import app
import os
@app.task(name='mlopenapp.tasks.make_venv')
def create_venv(pipeline_name):
    ''' Task to create a virtual environment for a pipeline in the background
        :param pipeline_name: The name of the control file associated with the pipeline.
    '''
    #get Pipeline object
    context = {}
    context["data"] = Pipeline.objects.get(control = pipeline_name)
    if not(os.path.exists('mlopenapp/venv/')):
        os.makedirs('mlopenapp/venv/')
    path_to_venv = 'mlopenapp/venv/' + pipeline_name
    path_to_reqs = 'mlopenapp/pipelines/' + pipeline_name + '/requirements.txt'
    
    #create venv for the pipeline 
    subprocess.call(['sh','mlopenapp/makevenv.sh', path_to_venv, path_to_reqs])
    
    #add path of venv to the model
    context["data"].venv = 'mlopenapp/venv/' + pipeline_name
    context["data"].save()
    return True

@app.task(name='mlopenapp.tasks.test_task_1')
def test_task_1(x, y):
    for i in range(0, 10):
        x = x + i
    return x

@app.task(name='mlopenapp.tasks.test_task_2')
def test_task_2(x, y):
    print("This is a test task")
    return True