import os
import os.path
import importlib.util
from django import forms
from mlopenapp.forms import PipelineSelectForm
from django.views.generic.base import TemplateView
from django.views.generic.edit import FormView
from django.http import JsonResponse
from ..models import MLPipeline as Pipeline
import os
from .. import constants
from ..utils import params_handler
import subprocess
import pickle
from os.path import exists



class PipelineView(TemplateView, FormView):
    template_name = "pipelines.html"
    form_class = PipelineSelectForm
    success_url = '/pipelines/'
    relative_url = "pipelines"
    CHOICES = [(0, 'Run Pipeline'),
               (1, 'Train Model')]

    def get_form(self, form_class=PipelineSelectForm):
        form = super().get_form(form_class)
        form.fields["type"] = forms.ChoiceField(choices=self.CHOICES, initial=0, required=False)
        return form

    def form_invalid(self, form):
        if self.request.is_ajax():
            clean_data = form.cleaned_data.copy()
            data = self.request.POST.get("select_pipeline", False)
            if data:
                try:
                    pipeline = self.request.POST.get("pipeline", False)
                    pipeline = Pipeline.objects.filter(id=int(pipeline)).first()
                    spec = importlib.util.spec_from_file_location(pipeline.control,
                                                                  os.path.join(
                                                                      constants.CONTROL_DIR,
                                                                      str(pipeline.control) + '.py'))
                    control = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(control)
                    type = self.request.POST.get("type", False)
                    type = True if type and int(type) == 0 else False
                    print("TYPE IS " + str(type))
                    params = control.get_params(type)
                    userform = params_handler.get_params_form(params)
                    return self.update_attrs(userform.as_table())
                except Exception as e:
                    return self.update_attrs("")
            if "pipelines" in clean_data:
                return self.update(clean_data)
            else:
                return JsonResponse({
                    "status": "false",
                    "messages": form.errors
                }, status=400)
        return self.render_to_response(self.get_context_data(form=form))

    def form_valid(self, form):
        if self.request.is_ajax():
            print("AH IT'S VALID")
            clean_data = form.cleaned_data.copy()
            data = self.request.POST.get("select_pipeline", False)
            if data:
                userform = forms.Form()
                userform.fields["sth"] = forms.CharField()
                return self.render_to_response(
                    self.get_context_data(form=form, userform="oooo"))
            if "pipelines" in clean_data:
                return self.update(clean_data)
            else:
                return JsonResponse({
                    "status": "false",
                    "messages": form.errors
                }, status=400)
        return self.render_to_response(self.get_context_data(form=form))

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["title"] = "Pipelines"
        context['template'] = "pipelines.html"
        if "userform" in kwargs:
            context["userform"] = kwargs["userform"]
        return context

    def update_attrs(self, userform):
        ret = {"userform": userform}
        return JsonResponse(ret, safe=False)

    def update(self, clean_data):
        inpt = clean_data['input']
        inpt = inpt.file if inpt else None

        pipeline = clean_data['pipelines']
        ret = []
        params = dict(self.request.POST)
        for name in ['type', 'pipelines', 'input']:
            params.pop(name, None)
        for name, param in params.items():
            if isinstance(param, list) and len(param) == 1:
                params[name] = param[0]
        try:
            if(pipeline.venv != ''):
                path_to_venv = 'mlopenapp/venv/' + pipeline.control
                subprocess.call(['sh','startvenv.sh', path_to_venv, 
                        pipeline.control,clean_data["type"],str(clean_data['input']),"params"])
                filename = constants.FILE_DIRS['graphs'] + '/' + pipeline.control + '.pkl'
                if(exists(filename)):
                    output = open(filename, 'rb')
                    ret = pickle.load(output)
                    output.close()
                else:
                    ret = {'error': True,
                    'error_msg': "Failed to create graph file."}
            else:
                ret = {"train": "Initializing virtual envoronment. Please try again later."}

        except Exception as e:
            ret = {'error': True,
            'error_msg': "There was a problem during the excecution of your pipeline.",
            'error_info': str(e)}
        return JsonResponse(ret, safe=False)
