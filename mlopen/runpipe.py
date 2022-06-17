
#from background_task import backgroundimport django
import django
django.setup()
from mlopenapp.utils import io_handler as io
import os
from mlopenapp import constants
import importlib.util
import sys
from mlopenapp.models.models import MLPipeline as Pipeline
from mlopenapp.models.files import InputFile as File
#args: pipeline.control, clean_data["type"], inpt, params
def run():
    
    pipeline = Pipeline.objects.get(control = sys.argv[1] )
    inpt = File.objects.get(name = sys.argv[3])
    inpt = inpt.file if inpt else None
    
    spec = importlib.util.spec_from_file_location(sys.argv[1],
                                                      os.path.join(constants.CONTROL_DIR,
                                                                   str(sys.argv[1]) + '.py'))
    control = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(control)
    print("Imported control file: ", control)
    ret = {}
    try:
        if sys.argv[2] == "0":
            print('INPUT IS 0 - RUNNING MODEL')
            model = None
            args = {}
            pipeline_ret = io.load_pipeline(pipeline)
            #print("Pipeline ret: ", pipeline_ret)
            if pipeline_ret:
                model = pipeline_ret[0]
                args = pipeline_ret[1]
                params = []
                preds = control.run_pipeline(inpt, model, args, params)
                if "graphs" in preds and preds["graphs"] not in [None, ""]:
                    if type(preds["graphs"]) is not list:
                        preds["graphs"] = [preds["graphs"]]
                print("done with graphs")
                ret = preds
        else:
                print('INPUT IS 1 - TRAINING MODEL')
                control.train(inpt)
                ret = {"train": "Training completed! You may now run the " + str(pipeline) + " pipeline."}

    except Exception as e:
            ret = {'error': True,
            'error_msg': "There was a problem during the excecution of your pipeline.",
            'error_info': str(e)}
    io.save_graphs(ret, pipeline.control)
    
    