# from gluonts_forecasts.utils import assert_time_column_valid

#     def create_gluon_dataset(self, remove_length=None):
#         length = -remove_length if remove_length else None
#         multivariate_timeseries = []
#         if self.timeseries_identifiers_names:
#             periods = None
#             for i, (identifiers_values, identifiers_df) in enumerate(self.training_df.groupby(self.timeseries_identifiers_names)):
#                 assert_time_column_valid(identifiers_df, self.time_column_name, self.frequency, periods=periods)
#                 if i == 0:
#                     periods = len(identifiers_df.index)
#                 multivariate_timeseries += self._create_gluon_multivariate_timeseries(identifiers_df, length, identifiers_values=identifiers_values)
#         else:
#             assert_time_column_valid(self.training_df, self.time_column_name, self.frequency)
#             multivariate_timeseries += self._create_gluon_multivariate_timeseries(self.training_df, length)
#         return ListDataset(multivariate_timeseries, freq=self.frequency)

#     def _create_gluon_multivariate_timeseries(self, df, length, identifiers_values=None):
#         multivariate_timeseries = []
#         for target_column_name in self.target_columns_names:
#             multivariate_timeseries.append(self._create_gluon_univariate_timeseries(df, target_column_name, length, identifiers_values))
#         return multivariate_timeseries

#     def _create_gluon_univariate_timeseries(self, df, target_column_name, length, identifiers_values=None):
#         """ create dictionary for one timeseries and add extra features and identifiers if specified """
#         univariate_timeseries = {
#             'start': df[self.time_column_name].iloc[0],
#             'target': df[target_column_name].iloc[:length].values,
#             'target_name': target_column_name,
#             'time_column_name': self.time_column_name
#         }
#         if self.external_features_columns_names:
#             univariate_timeseries['feat_dynamic_real'] = df[self.external_features_columns_names].iloc[:length].values.T
#             univariate_timeseries['feat_dynamic_real_columns_names'] = self.external_features_columns_names
#         if identifiers_values:
#             if len(self.timeseries_identifiers_names) > 1:
#                 identifiers_map = {self.timeseries_identifiers_names[i]: identifier_value for i, identifier_value in enumerate(identifiers_values)}
#             else:
#                 identifiers_map = {self.timeseries_identifiers_names[0]: identifiers_values}
#             univariate_timeseries['identifiers'] = identifiers_map
#         return univariate_timeseries