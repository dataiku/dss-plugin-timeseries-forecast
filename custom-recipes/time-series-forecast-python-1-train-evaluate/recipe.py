# -*- coding: utf-8 -*-
import pandas as pd
from dataiku.customrecipe import get_recipe_config
from datetime import datetime
from plugin_io_utils import get_models_parameters, save_dataset
from dku_timeseries.global_models import GlobalModels
from dku_timeseries.gluon_ts_dataset import GlutonTSDataset
from plugin_config_loading import load_training_config, load_evaluation_config

config = get_recipe_config()

global_params = load_training_config(config)
eval_params = load_evaluation_config(config) # merge these two
version_name = datetime.now().strftime('%Y-%m-%dT%H-%M-%S-%f')[:-3]

models_parameters = get_models_parameters(config)

glutonts_training_dataset = GlutonTSDataset(
    dataset_name=global_params['input_dataset_name'],
    time_column_name=global_params['time_column_name'],
    target_column_name=global_params['target_column_name'],
    frequency=global_params['frequency']
)# todo modify that one to integrate extra columns and multiple target columns
save_dataset(
    dataset_name=global_params['input_dataset_name'],
    time_column_name=global_params['time_column_name'],
    target_column_name=global_params['target_column_name'],
    model_folder=global_params['model_folder'],
    version_name=version_name
)

training_df = 

global_models = GlobalModels(global_params, models_parameters, training_df)
global_models.init_all_models()

global_models.evaluate_all(params['evaluation_strategy'])

global_models.fit_all()

# predictors_error = global_models.evaluate_all(eval_params, glutonts_training_dataset)
df = pd.json_normalize(predictors_error)
eval_params['evaluation_dataset'].write_schema_from_dataframe(df)
writer = eval_params['evaluation_dataset'].get_writer()
writer.write_dataframe(df)

global_models.save_all(version_name=version_name)

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