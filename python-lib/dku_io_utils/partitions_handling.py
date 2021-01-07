import dataiku


TIME_DIMENSION_PATTERNS = {"YEAR": "%Y", "MONTH": "%M", "DAY": "%D", "HOUR": "%H"}


def get_folder_partition_root(folder, is_input=False):
    """Retrieve the partition root path using a dataiku.Folder.

    Args:
        folder (dataiku.Folder): Input or output folder of the recipe used to retrieve the partition path pattern.
        is_input:  True if the folder must be considered as a input, False if output

    Returns:
        Partition path or None if folder is not partitioned.
    """
    folder_id = folder.get_id()
    input_id = folder_id if is_input else None
    dku_flow_variables = dataiku.get_flow_variables()
    client = dataiku.api_client()
    project = client.get_project(dataiku.default_project_key())
    folder = project.get_managed_folder(folder_id)
    folder_config = folder.get_definition()
    partitioning_config = folder_config.get("partitioning")
    if not partitioning_config:
        return ""
    file_path_pattern = partitioning_config.get("filePathPattern", None)
    dimensions, types = get_dimensions(partitioning_config)
    partitions = get_partitions(dku_flow_variables, dimensions, input_id=input_id)
    file_path = complete_file_path_pattern(file_path_pattern, partitions, dimensions, types)
    file_path = complete_file_path_time_pattern(dku_flow_variables, file_path, input_id=input_id)
    return file_path


def get_dimensions(partitioning_config):
    """Retrieve the list of partition dimension names.

    Args:
        partitioning_config (dict): Dictionary of partitioning variables.

    Returns:
        List of dimensions.
    """
    dimensions_dict = partitioning_config.get("dimensions")
    dimensions = []
    types = []
    for dimension in dimensions_dict:
        dimensions.append(dimension.get("name"))
        types.append(dimension.get("type"))
    return dimensions, types


def get_partitions(dku_flow_variables, dimensions, input_id=None):
    """Retrieve the list of partition values corresponding to the partition dimensions.

    Args:
        dku_flow_variables (dict): Dictionary of flow variables for a project.
        dimensions (list): List of partition dimensions.
        input_id (str): folder id if the folder is an input, None for output folder

    Raises:
        ValueError: If a 'DKU_DST_$DIMENSION' is not in dku_flow_variables.

    Returns:
        List of partitions.
    """
    partitions = []
    for dimension in dimensions:
        partition = get_dimension_value_from_flow_variables(dku_flow_variables, input_id, dimension)
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
        partitions = fix_date_elements_folder_path(partitions, types)
        return "/".join(partitions)
    file_path = file_path_pattern.replace(".*", "")
    for partition, dimension in zip(partitions, dimensions):
        file_path = file_path.replace(f"%{{{dimension}}}", partition)
    return file_path


def fix_date_elements_folder_path(partitions, types):
    """ Replace the '-' separator in time dimension with '/' so they can be used in folder paths """
    fixed_partitions = []
    for partition, type in zip(partitions, types):
        if type == "time":
            fixed_partitions.append(partition.replace("-", "/"))
        else:
            fixed_partitions.append(partition)
    return fixed_partitions


def complete_file_path_time_pattern(dku_flow_variables, file_path_pattern, input_id=None):
    """Fill the placeholders of the partition path pattern for the time dimensions with the right partition values.

    Args:
        dku_flow_variables (dict): Dictionary of flow variables for a project.
        file_path_pattern (str)
        input_id (str): folder id if the folder is an input, None if for output folder

    Returns:
        File path prefix.
    """
    file_path = file_path_pattern
    for time_dimension in TIME_DIMENSION_PATTERNS:
        time_value = get_dimension_value_from_flow_variables(dku_flow_variables, input_id, time_dimension)
        if time_value is not None:
            time_pattern = TIME_DIMENSION_PATTERNS.get(time_dimension)
            file_path = file_path.replace(time_pattern, time_value)
    return file_path


def get_dimension_value_from_flow_variables(dku_flow_variables, input_id, dimension):
    origin = "DST" if input_id is None else "SRC_{}".format(input_id)
    return dku_flow_variables.get("DKU_{}_{}".format(origin, dimension))
