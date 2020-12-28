import dataiku


TIME_DIMENSION_PATTERNS = {"DKU_DST_YEAR": "%Y", "DKU_DST_MONTH": "%M", "DKU_DST_DAY": "%D", "DKU_DST_HOUR": "%H"}


def get_partition_root(dataset):
    """Retrieve the partition root path using a dataiku.Dataset.

    Args:
        dataset (dataiku.Dataset): Input or output dataset of the recipe used to retrieve the partition path pattern.

    Returns:
        Partition path or None if dataset is not partitioned.
    """
    dku_flow_variables = dataiku.get_flow_variables()
    file_path_pattern = dataset.get_config().get("partitioning").get("filePathPattern", None)

    dimensions, types = get_dataset_dimensions(dataset)
    partitions = get_partitions(dku_flow_variables, dimensions)
    file_path = complete_file_path_pattern(file_path_pattern, partitions, dimensions, types)
    file_path = complete_file_path_time_pattern(dku_flow_variables, file_path)

    return file_path


def get_folder_partition_root(folder):
    """Retrieve the partition root path using a dataiku.Folder.

    Args:
        dataset (dataiku.Dataset): Input or output dataset of the recipe used to retrieve the partition path pattern.

    Returns:
        Partition path or None if dataset is not partitioned.
    """
    folder_id = folder.get_id()
    dku_flow_variables = dataiku.get_flow_variables()
    client = dataiku.api_client()
    project = client.get_project(dataiku.default_project_key())
    folder = project.get_managed_folder(folder_id)
    folder.get_definition()["partitioning"]
    folder_config = folder.get_definition()
    file_path_pattern = folder_config.get("partitioning").get("filePathPattern", None)
    dimensions, types = get_dimensions(folder_config)
    partitions = get_partitions(dku_flow_variables, dimensions)
    file_path = complete_file_path_pattern(file_path_pattern, partitions, dimensions, types)
    file_path = complete_file_path_time_pattern(dku_flow_variables, file_path)
    return file_path


def get_dataset_dimensions(dataset):
    dataset_config = dataset.get_config()
    return get_dimensions(dataset_config)


def get_dimensions(dataset_config):
    """Retrieve the list of partition dimension names.

    Args:
        dataset (dataiku.Dataset)

    Returns:
        List of dimensions.
    """
    dimensions_dict = dataset_config.get("partitioning").get("dimensions")
    dimensions = []
    types = []
    for dimension in dimensions_dict:
        dimensions.append(dimension.get("name"))
        types.append(dimension.get("type"))
    return dimensions, types


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
        partition = dku_flow_variables.get(f"DKU_DST_{dimension}")
        if partition is None:
            raise ValueError(
                f"Partition dimension '{dimension}' not found in output. Please make sure your output has the same partition dimensions as your input."
            )
        partitions.append(partition)
    return partitions


def complete_file_path_pattern(file_path_pattern, partitions, dimensions, types):
    """Fill the placeholders of the partition path pattern for the discrete dimensions with the right partition values.

    Args:
        file_path_pattern (str)
        partitions (list): List of partition values corresponding to the partition dimensions.
        dimensions (list): List of partition dimensions.

    Returns:
        File path prefix. Time dimensions pattern are not filled.
    """

    if file_path_pattern is None:
        # Probably SQL dataset
        partitions = fix_date_elements_folder_path(partitions, types)
        return "/".join(partitions)
    file_path = file_path_pattern.replace(".*", "")
    for partition, dimension in zip(partitions, dimensions):
        file_path = file_path.replace(f"%{{{dimension}}}", partition)
    return file_path


def fix_date_elements_folder_path(partitions, types):
    """ Replace the '-' separator in time dimension on SQL-type dataset by '/' so they can be used in folder paths """
    fixed_partitions = []
    for partition, type in zip(partitions, types):
        if type == "time":
            fixed_partitions.append(partition.replace("-", "/"))
        else:
            fixed_partitions.append(partition)
    return fixed_partitions


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
