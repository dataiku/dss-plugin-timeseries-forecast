from gluonts.dataset.common import ListDataset
import dataiku
import pandas as pd


class GlutonTSDataset():
    def __init__(self, dataset_name, time_column_name, target_column_name, frequency):
        dataset = dataiku.Dataset(dataset_name)
        dataset_df = dataset.get_dataframe()   # todo : move this outside
        initial_date = dataset_df[time_column_name].iloc[0]
        start = pd.Timestamp(initial_date, freq=frequency).tz_localize(None)
        self.list_dataset = ListDataset(
            #  [{"start": dataset_df.index[0], "target": dataset_df.get(target_column_name)}],
            [{"start": start, "target": dataset_df.get(target_column_name)}],
            freq=frequency
        )
        self.target_column_name = target_column_name
        for row in self.list_dataset:
            print("ALX:row={}".format(row))

    def get_list_dataset(self):
        return self.list_dataset
