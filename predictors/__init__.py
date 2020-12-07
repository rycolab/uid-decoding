import os
import inspect
import importlib
from .core import Predictor

PREDICTOR_REGISTRY = {}

predictor_dir = os.path.dirname(__file__)
for file in os.listdir(predictor_dir):
    path = os.path.join(predictor_dir, file)
    if not file.startswith('_') and not file.startswith('.') and file.endswith('.py'):
        model_name = file[:file.find('.py')]
        module = importlib.import_module('predictors.' + model_name)
        clsmembers = inspect.getmembers(module, inspect.isclass)
        for name, _cls in clsmembers:
        	if issubclass(_cls, Predictor) and not _cls == Predictor:
        		if not hasattr(_cls, 'name'):
        			raise ValueError("All predictor classes must have `name` attribute. Culprit: {}".format(name))
        		else:
        			PREDICTOR_REGISTRY[_cls.name] = _cls
