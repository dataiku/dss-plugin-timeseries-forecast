from dataiku.customrecipe import get_input_names_for_role, get_output_names_for_role
import dataiku
from dkulib.dku_config import DkuConfig


class DkuFileManager(DkuConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def add_file(self, side, type_, role, **kwargs):
        file = DkuFileManager._retrieve_file_from_dss(side, type_, role)
        self.add_param(name=role, value=file, **kwargs)

    def add_input_folder(self, role, required=True):
        self.add_file("input", "folder", role, required=required)

    def add_output_folder(self, role, required=True):
        self.add_file("output", "folder", role, required=required)

    def add_input_dataset(self, role, required=True):
        self.add_file("input", "dataset", role, required=required)

    def add_output_dataset(self, role, required=True):
        self.add_file("output", "dataset", role, required=required)

    @staticmethod
    def _retrieve_file_from_dss(side, type_, role):
        dku_func = get_input_names_for_role if side == "input" else get_output_names_for_role
        dku_type = dataiku.Folder if type_ == "folder" else dataiku.Dataset
        roles = dku_func(role)
        return dku_type(roles[0]) if roles else None

    @staticmethod
    def write_to_folder(folder, file_path, content):
        with folder.get_writer(file_path) as w:
            w.write(content.encode())
