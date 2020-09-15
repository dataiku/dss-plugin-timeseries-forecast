class GlobalParams():
    # see tools in plugin_io_utils
    models_names = ["naive", "simplefeedforward", "deepfactor", "deepar", "lstnet"]
    models_params_map = {"model_name":["param1", "param2"]}
    global_params

    def __init__(self, config):
        self.model_param = {"model_name": "model_param"}

        self._check()

    def get_model_params(self, model_name):
        ret = global_params
        ret.update(self.model_param.get(model_name))
        return ret
    
    def get_global_model_params(self):
        return

    def _check(self):
        return
