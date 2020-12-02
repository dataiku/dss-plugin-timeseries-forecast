import io
import dill as pickle
import pandas as pd
import json
import gzip
from safe_logger import SafeLogger

logger = SafeLogger("Forecast plugin")


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
    logger.info("Loading {} from folder".format(path))
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
    logger.info("Saving {} to folder".format(path))
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


def set_column_description(output_dataset, column_description_dict, input_dataset=None, to_lowercase=False):
    """Set column descriptions of the output dataset based on a dictionary of column descriptions

    Retains the column descriptions from the input dataset if the column name matches.

    Args:
        output_dataset: Output dataiku.Dataset instance
        column_description_dict: Dictionary holding column descriptions (value) by column name (key)
        input_dataset: Optional input dataiku.Dataset instance
            in case you want to retain input column descriptions
        to_lowercase: Convert to lowercase column names that are in column_description_dict
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
        if to_lowercase and output_col_name in column_description_dict:
            output_col_info["name"] = output_col_name.lower()
    output_dataset.write_schema(output_dataset_schema)
