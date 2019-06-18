import configs
from pprint import pprint
import jsonpickle
import jsonpickle.ext.numpy
import json

import jsonpickle.ext.numpy as jsonpickle_numpy
jsonpickle_numpy.register_handlers()


def test_view():
    pprint(configs.default.view())

def test_jsonpickle():
    jsonpickle.ext.numpy.register_handlers()
    config = configs.default
    #print(json.dumps(json.loads(jsonpickle.encode(configs.default, make_refs=False)), indent=4, sort_keys=True))
    p = jsonpickle.pickler.Pickler()
    json_dict = p.flatten(configs.default)
    pprint(json_dict)

    u = jsonpickle.unpickler.Unpickler()


    config_out = u.restore(json_dict)
    print(config_out)
