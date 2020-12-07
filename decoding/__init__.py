import os
import inspect
import importlib
from .core import Decoder

DECODER_REGISTRY = {}

decoders_dir = os.path.dirname(__file__)
for file in os.listdir(decoders_dir):
    path = os.path.join(decoders_dir, file)
    if not file.startswith('_') and not file.startswith('.') and file.endswith('.py'):
        model_name = file[:file.find('.py')]
        module = importlib.import_module('decoding.' + model_name)
        clsmembers = inspect.getmembers(module, inspect.isclass)
        for name, _cls in clsmembers:
        	if issubclass(_cls, Decoder) and not _cls == Decoder:
        		if not hasattr(_cls, 'name'):
        			raise ValueError("All decoder classes must have `name` attribute. Culprit: {}".format(name))
        		else:
        			DECODER_REGISTRY[_cls.name] = _cls
