import io
import os
import dill as pickle
import dataiku
import pandas as pd
import json
import gzip
from constants import TIME_DIMENSION_PATTERNS
from safe_logger import SafeLogger

logging = SafeLogger("Forecast plugin")


def read_from_folder(folder, path, object_type):
    """Read object from folder/path using the reading method correponding to object_type.

    Args:
        folder (dataiku.Folder)
        path (str): Path within the folder.
        object_type (str): Type of object to read. Must be one of ('pickle', 'pickle.gz', 'csv', 'csv.gz').

    Raises:
        ValueError: No file were found at the requested path.
        ValueError: Object type is not supported.

    Returns:
        Object based on the requested type.
    """
    logging.info("Loading {}".format(os.path.join(folder.get_path(), path)))
    if not folder.get_path_details(path=path)["exists"]:
        raise ValueError("File at path {} doesn't exist in folder {}".format(path, folder.get_info()["name"]))
    with folder.get_download_stream(path) as stream:
        if object_type == "pickle":
            return pickle.loads(stream.read())
        if object_type == "pickle.gz":
            with gzip.GzipFile(fileobj=stream) as fgzip:
                return pickle.loads(fgzip.read())
        elif object_type == "csv":
            data = io.StringIO(stream.read().decode())
            return pd.read_csv(data)
        elif object_type == "csv.gz":
            with gzip.GzipFile(fileobj=stream) as fgzip:
                return pd.read_csv(fgzip)
        else:
            raise ValueError("Can only read objects of type ['pickle', 'pickle.gz', 'csv', 'csv.gz'] from folder, not '{}'".format(object_type))


def write_to_folder(object_to_save, folder, path, object_type):
    """Write object_to_save to folder/path using the writing method correponding to object_type.

    Args:
        object_to_save (any type supported by the object type): Object that will be saved in the folder.
        folder (dataiku.Folder)
        path (str): Path within the folder.
        object_type (str): Type of object to save. Must be one of ('pickle', 'pickle.gz', 'csv', 'csv.gz').

    Raises:
        ValueError: Object type is not supported.
    """
    logging.info("Saving {}".format(os.path.join(folder.get_path(), path)))
    with folder.get_writer(path) as writer:
        if object_type == "pickle":
            writeable = pickle.dumps(object_to_save)
            writer.write(writeable)
        elif object_type == "pickle.gz":
            writeable = pickle.dumps(object_to_save)
            with gzip.GzipFile(fileobj=writer, mode="wb", compresslevel=9) as fgzip:
                fgzip.write(writeable)
        elif object_type == "json":
            writeable = json.dumps(object_to_save).encode()
            writer.write(writeable)
        elif object_type == "csv":
            writeable = object_to_save.to_csv(sep=",", na_rep="", header=True, index=False).encode()
            writer.write(writeable)
        elif object_type == "csv.gz":
            with gzip.GzipFile(fileobj=writer, mode="wb") as fgzip:
                fgzip.write(object_to_save.to_csv(index=False).encode())
        else:
            raise ValueError("Can only write objects of type ['pickle', 'pickle.gz', 'json', 'csv', 'csv.gz'] to folder, not '{}'".format(object_type))


def set_column_description(output_dataset, column_description_dict, input_dataset=None):
    """Set column descriptions of the output dataset based on a dictionary of column descriptions

    Retains the column descriptions from the input dataset if the column name matches.

    Args:
        output_dataset: Output dataiku.Dataset instance
        column_description_dict: Dictionary holding column descriptions (value) by column name (key)
        input_dataset: Optional input dataiku.Dataset instance
            in case you want to retain input column descriptions
    """
    output_dataset_schema = output_dataset.read_schema()
    input_dataset_schema = []
    input_columns_names = []
    if input_dataset is not None:
        input_dataset_schema = input_dataset.read_schema()
        input_columns_names = [col["name"] for col in input_dataset_schema]
    for output_col_info in output_dataset_schema:
        output_col_name = output_col_info.get("name", "")
        output_col_info["comment"] = column_description_dict.get(output_col_name)
        if output_col_name in input_columns_names:
            matched_comment = [input_col_info.get("comment", "") for input_col_info in input_dataset_schema if input_col_info.get("name") == output_col_name]
            if len(matched_comment) != 0:
                output_col_info["comment"] = matched_comment[0]
    output_dataset.write_schema(output_dataset_schema)


def get_partition_root(dataset):
    """Retrieve the partition root path using a dataiku.Dataset.

    Args:
        dataset (dataiku.Dataset): Input or output dataset of the recipe used to retrieve the partition path pattern.

    Returns:
        Partition path or None if dataset is not partitioned.
    """
    dku_flow_variables = dataiku.get_flow_variables()
    file_path_pattern = dataset.get_config().get("partitioning").get("filePathPattern", None)
    if file_path_pattern is None:
        return None
    dimensions = get_dimensions(dataset)
    partitions = get_partitions(dku_flow_variables, dimensions)
    file_path = complete_file_path_pattern(file_path_pattern, partitions, dimensions)
    file_path = complete_file_path_time_pattern(dku_flow_variables, file_path)
    return file_path


def get_dimensions(dataset):
    """Retrieve the list of partition dimension names.

    Args:
        dataset (dataiku.Dataset)

    Returns:
        List of dimensions.
    """
    dimensions_dict = dataset.get_config().get("partitioning").get("dimensions")
    dimensions = []
    for dimension in dimensions_dict:
        dimensions.append(dimension.get("name"))
    return dimensions


def get_partitions(dku_flow_variables, dimensions):
    """Retrieve the list of partition values corresponding to the partition dimensions.

    Args:
        dku_flow_variables (dict): Dictionary of flow variables for a project.
        dimensions (list): List of partition dimensions.

    Raises:
        ValueError: If a 'DKU_DST_$DIMENSION' is not in dku_flow_variables.

    Returns:
        List of partitions.
    """
    partitions = []
    for dimension in dimensions:
        partition = dku_flow_variables.get("DKU_DST_{}".format(dimension))
        if partition is None:
            raise ValueError("Partition dimension '{}' not found in output. Make sure the output has the same partitioning as the input".format(dimension))
        partitions.append(partition)
    return partitions


def complete_file_path_pattern(file_path_pattern, partitions, dimensions):
    """Fill the placeholders of the partition path pattern for the discrete dimensions with the right partition values.

    Args:
        file_path_pattern (str)
        partitions (list): List of partition values corresponding to the partition dimensions.
        dimensions (list): List of partition dimensions.

    Returns:
        File path prefix. Time dimensions pattern are not filled.
    """
    file_path = file_path_pattern.replace(".*", "")
    for partition, dimension in zip(partitions, dimensions):
        file_path = file_path.replace("%{{{}}}".format(dimension), partition)
    return file_path


def complete_file_path_time_pattern(dku_flow_variables, file_path_pattern):
    """Fill the placeholders of the partition path pattern for the time dimensions with the right partition values.

    Args:
        dku_flow_variables (dict): Dictionary of flow variables for a project.
        file_path_pattern (str)

    Returns:
        File path prefix.
    """
    file_path = file_path_pattern
    for time_dimension in TIME_DIMENSION_PATTERNS:
        time_value = dku_flow_variables.get(time_dimension)
        if time_value is not None:
            time_pattern = TIME_DIMENSION_PATTERNS.get(time_dimension)
            file_path = file_path.replace(time_pattern, time_value)
    return file_path
