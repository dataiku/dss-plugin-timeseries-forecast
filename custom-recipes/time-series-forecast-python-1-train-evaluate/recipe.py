# -*- coding: utf-8 -*-
import pandas as pd
import dataiku
from dataiku.customrecipe import get_recipe_config
from datetime import datetime
from plugin_io_utils import get_models_parameters, save_dataset
from dku_timeseries.global_models import GlobalModels
from plugin_config_loading import load_training_config

config = get_recipe_config()

global_params = load_training_config(config)
version_name = datetime.now().strftime('%Y-%m-%dT%H-%M-%S-%f')[:-3]

models_parameters = get_models_parameters(config)

# TODO save with compression
save_dataset(
    dataset_name=global_params['input_dataset_name'],
    time_column_name=global_params['time_column_name'],
    target_columns_names=global_params['target_columns_names'],
    model_folder=global_params['model_folder'],
    version_name=version_name
)

training_dataset = dataiku.Dataset(global_params['input_dataset_name'])
training_df = training_dataset.get_dataframe()

global_models = GlobalModels(
    target_columns_names=global_params['target_columns_names'],
    time_column_name=global_params['time_column_name'],
    frequency=global_params['frequency'],
    model_folder=global_params['model_folder'],
    epoch=global_params['epoch'],
    models_parameters=models_parameters,
    prediction_length=global_params['prediction_length'],
    training_df=training_df,
    forecast=global_params.get('evaluation_forecasts')
)  # todo : integrate external features and multiple target columns
global_models.init_all_models()

df = global_models.evaluate_all(global_params['evaluation_strategy'])

global_models.fit_all()

global_params['evaluation_dataset'].write_schema_from_dataframe(df)
writer = global_params['evaluation_dataset'].get_writer()
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