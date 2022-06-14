from mlopenapp.models.models import MLPipeline as Pipeline
import os
from django.http import JsonResponse
import subprocess
from mlopenapp.models.models import MLPipeline as Pipeline
from mlopen.celery import app

@app.task(name='mlopenapp.tasks.make_venv')
def create_venv(pipeline_name):
    print("Making venv..")
    context = {}
    #print("making pipeline for pipeline with name", pipeline_name)
    context["data"] = Pipeline.objects.get(control = 'yolo-control')

    #print('Pipeline name:', context["data"].control)
    #makevenv = 'python3 -m venv mlopen/mlopenapp/venv/' + pipeline_name
    path_to_venv = 'mlopenapp/venv/' + pipeline_name
    path_to_reqs = 'mlopenapp/pipelines/' + pipeline_name + '/requirements.txt'
    #os.system(makevenv)
    subprocess.call(['sh','mlopenapp/makevenv.sh', path_to_venv, path_to_reqs])
    context["data"].venv = 'mlopenapp/venv/' + pipeline_name
    context["data"].save()
    print('venv done')
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