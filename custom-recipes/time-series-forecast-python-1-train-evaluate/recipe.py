# -*- coding: utf-8 -*-
import dataiku
from dataiku.customrecipe import get_input_names_for_role, get_recipe_config, get_output_names_for_role
from gluonts.dataset.common import ListDataset
from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer
from datetime import datetime
import os
from plugin_io_utils import get_models_parameters, get_estimator, save_forecasting_objects, evaluate_models
import pandas as pd

from eval_parameters import EvalParams
from global_params import GlobalParams
from global_models import GlobalModels

# TODO
config = get_recipe_config()
input_dataset_name = get_input_names_for_role('input_dataset')[0]
model_folder_name = get_output_names_for_role('model_folder')[0]
model_folder = dataiku.Folder(model_folder_name)
evaluation_dataset_name = get_output_names_for_role('evaluation_dataset')[0]
evaluation_dataset = dataiku.Dataset(evaluation_dataset_name)
evaluation_dataset_name = get_output_names_for_role('evaluation_dataset')[0]
target_column_name = config.get("target_column")
time_column_name = config.get("time_column")
external_feature_columns = config.get('external_feature_columns', [])
deepar_model_activated = config.get('deepar_model_activated', False)
time_granularity_unit = config.get('time_granularity_unit')
time_granularity_step = config.get('time_granularity_step', 1)
frequency = "{}{}".format(time_granularity_step, time_granularity_unit)

models_parameters = get_models_parameters(config)



glutonts_dataset = GlutonTSDataset(recipe_config, input_dataset)

eval_params = EvalParams(recipe_config) # backtest, forecasting horizon, num_samples, ...

global_params = GlobalParams(recipe_config)

global_models = GlobalModels(global_params)
global_models.init_all_models()

global_models.fit_all(glutonts_dataset)

global_models.eval_all(eval_params, glutonts_dataset)

global_models.save_all(path)



"""
class GlobalParams():
    # see tools in plugin_io_utils
    models_names = ["naive", "simplefeedforward", "deepfactor", "deepar", "lstnet"]
    models_params_map = {"model_name":["param1", "param2"]}
    global_params
    def __init__(config):
        self.model_param = {"model_name": "model_param"}

        self._check():

    def get_model_params(self, model_name):
        ret = global_params
        ret.update(self.model_param.get(model_name))
        return ret
    
    def get_global_model_params(self):

    def _check(self):
        
        
class GlobalModels():
    def __init__(global_params):
        self.global_params = global_params

        self.model_names = []

    def init_all_models(self):
        self.models = []
        for model_name in self.global_params.models_names:
            self.models.append(SingleModel(model_name, self.global_params.get_model_params(model_name), self.global_params.get_global_model_params()))


    def fit_all(self, gluonts_dataset):
        for model in self.models:
            model.fit(gluonts_dataset)

    def evaluate_all(self, eval_params, glutonts_dataset):
        for model in self.models:
            model.evaluate(eval_params, glutonts_dataset)

    def save_all(path):
        for model in self.models:
            model.save(path="{}/{}".format(path, model_name))
        save(dataset) # csv with results

    def prediction(model_name):
        
    def load(path):
        dataset = load(dataset)
        best_model = find_best_model(dataset)
        model = SingleModel()
        model.load(path, best_model)
"""


# In prediction recipe
results_dataset = load_dataset(input_folder)
best_model_name = find_best_model(results_dataset)
model = load_model(best_model_name) # => SingleModel()


    
"""
class SingleModel():
    def __init__(model_name, model_params, global_models_params):
        

    def fit(self, gluonts_dataset):

    def evaluate(self, eval_params, glutonts_dataset):

    def save(self):
"""



print("ALX:config={}".format(config))

input_dataset = dataiku.Dataset(input_dataset_name)
input_dataset_df = input_dataset.get_dataframe()
data_list = []
"""
for index, row in input_dataset_df.iterrows():
    data = row[target_column_name]
    time = row[time_column_name]
    print("ALX: data={} / {}".format(data, time))
"""
print("ALX:before ListDataset")

training_data = ListDataset(
    [{"start": input_dataset_df.index[0], "target": input_dataset_df.get(target_column_name)}],
    freq=frequency
)
"""
if deepar_model_activated:
    estimator = DeepAREstimator(freq=frequency, prediction_length=12, trainer=Trainer(epochs=10))
    predictor = estimator.train(training_data=training_data)
    filename = "predictor_{}.pk".format(datetime.now().strftime('%Y-%m-%dT%H-%M-%S-%f')[:-3])
    with open(os.path.join(model_folder.get_path(), filename), 'wb') as predictor_file:
        pickle.dump(predictor, predictor_file)
"""

predictor_objects = {}
for model in models_parameters:
    print("ALX:model={}".format(model))
    estimator = get_estimator(model, models_parameters.get(model), freq=frequency, prediction_length=10, trainer=Trainer(epochs=10))
    if estimator is None:
        print("ALX:not imp yet")
        continue
    predictor = estimator.train(training_data=training_data)
    predictor_objects.update({model: predictor})

print("ALX:Saving objects")
forecasting_object = {
    "models": predictor_objects,
    "training_data": training_data,
    "target_column_name": target_column_name
}
save_forecasting_objects(model_folder.get_path(), datetime.now().strftime('%Y-%m-%dT%H-%M-%S-%f')[:-3], forecasting_object)

print("ALX:done")

##### EVALUATION STAGE #####

evaluation_strategy = config.get("evaluation_strategy", "split")
forecasting_horizon = config.get("forecasting_horizon", 1)

print("Evaluation stage starting with {} strategy...".format(evaluation_strategy))

predictors_error = evaluate_models(predictor_objects, training_data, evaluation_strategy=evaluation_strategy, forecasting_horizon=forecasting_horizon)
print("ALX:dir={}".format(dir(evaluation_dataset)))
df = pd.json_normalize(predictors_error)
evaluation_dataset.write_schema_from_dataframe(df)
writer = evaluation_dataset.get_writer()
writer.write_dataframe(df)


# Naive estimator is in fact 3 models
# kwargs by default could be made visible in the interface (when in expert mode)
# Trainer has it's own set of kwargs. 2 kwargs in the interface or key prefix ?

"""
ketchup du 
models.rdata
contient tous les models entraines et les données d'entrainement (archivage), parametres d'entrainement

2eme recipe 
normalement, output vide

python: 
interogation du model : même horizon qu'a l'entrainement pour la version python (!= que R)
"""

"""
objet instancié par model, avec fonction create json
model sauvé par repertoire 
"""