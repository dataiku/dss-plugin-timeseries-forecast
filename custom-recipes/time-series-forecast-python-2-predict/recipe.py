# -*- coding: utf-8 -*-
import dataiku
from dataiku.customrecipe import get_recipe_config, get_input_names_for_role, get_output_names_for_role
import os

# TODO
config = get_recipe_config()
model_folder_name = get_input_names_for_role('model_folder')[0]
model_folder = dataiku.Folder(model_folder_name)
output_dataset_name = get_output_names_for_role('output_dataset')[0]
output_dataset = dataiku.Dataset(output_dataset_name)
forecasting_horizon = config.get("forecasting_horizon", 1)

path = os.path.join(model_folder.get_path(), "versions")
print("ALX:path={}".format(path))
#model_directories = os.walk(path)
#model_directories = [f.path for f in os.scandir(path) if f.is_dir()]
model_directories = os.listdir(path)
model_directories.sort(reverse=True)
last_model_path = os.path.join(path, model_directories[0])

import dill as pickle
from gluonts.evaluation.backtest import make_evaluation_predictions

with open(os.path.join(last_model_path, "models.pk"), 'rb') as forecasting_file:
    forecasting_object = pickle.load(forecasting_file)
print("ALX:tadaaa")
"""
forecasting_object = {
    "models": predictor_objects,
    "training_data": training_data,
    "target_column_name": target_column_name
}
"""
predictor_objects = forecasting_object.get("models")
target_column_name = forecasting_object.get("target_column_name")
training_data = forecasting_object.get("training_data")
print("ALX:training_data={}".format(training_data))

for predictor in predictor_objects:
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=training_data,  # test dataset
        predictor=predictor_objects[predictor],  # predictor
        num_samples=forecasting_horizon,  # number of sample paths we want for evaluation
    )
    forecasts = list(forecast_it)
    tss = list(ts_it)
    print("ALX:tss[0].head()={}".format(tss[0].head()))  # <- contains the 5 first values, starting at epoch 0 + 4 days

    print(f"Number of sample paths: {forecasts[0].num_samples}")  # 4 sample path... = forecasting_horizon
    print(f"Dimension of samples: {forecasts[0].samples.shape}")  # 4 x 12 forecasting_horizon x estimators' prediction_length
    print(f"Start date of the forecast window: {forecasts[0].start_date}")
    print(f"Frequency of the time series: {forecasts[0].freq}")

    print("ALX:dir(forecasts)={}".format(dir(forecasts)))
    print("ALX:forecasts={}".format(forecasts))
    print("ALX:mean={}".format(forecasts[0].mean))  # <- 12 samples = estimators' prediction_length
    print("ALX:dir={}".format(dir(forecasts[0])))
    print("ALX:forecast_it={}".format(list(forecast_it)))
    for forecast in forecast_it:
        print("ALX:forecast={}".format(forecast))
    # training_data["orgin"] = "history"

print("ALX:training_data={}".format(list(training_data)[0].get("target")))
#df = training_data.assign(origin=['history'])
#output_dataset.write_schema_from_dataframe(df)
#writer = output_dataset.get_writer()
#writer.write_dataframe(df)
