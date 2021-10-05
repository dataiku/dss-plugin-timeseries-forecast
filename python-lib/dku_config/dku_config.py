from .dss_parameter import DSSParameter
from collections.abc import MutableMapping
from typing import Any, AnyStr


class DkuConfig(MutableMapping):
    """Mapping structure containing DSSParameter objects. It behaves as a dict with the following differences:
        - You can access elements with a dot structure (Example: dku_config.param1 or dku_config["param1"])
        - You can set an element with a dot structure (Example: dku_config.param1 = 123)
        - All objects stored are converted in DSSParameter
        - Accessing an element returns the value of the object DSSParameter

    Attributes:
        config(dict): Dict storing the DSSParameters
    """
    def __init__(self, local_vars: dict = None, local_prefix: AnyStr = '', **kwargs):
        """Initialization method for the DkuConfig class

        Args:
            local_vars(dict, optional): Dict containing vars fetched from project local variables. Default is {}
            local_prefix(str, optional): If project vars prefixed, write the prefix here, it will be added when
                searching for the var
            **kwargs: DSSParameters. Each key will be set as the parameter name and the values must be of type
                dict. These dicts must contain at least an attribute "value". For other attributes, see
                DSSParameter help.
        """
        object.__setattr__(self, 'config', {})
        object.__setattr__(self, 'local_vars', local_vars or {})
        object.__setattr__(self, 'local_prefix', local_prefix)
        if kwargs:
            for k, v in kwargs.items():
                if 'value' not in v:
                    raise ValueError('Each init kwargs must have a "value" field.')
                val = v.pop('value')
                self.add_param(name=k, value=val, **v)

    def add_param(self, name: AnyStr, value: Any = None, **kwargs):
        """Add a new DSSParameter to the config

        Args:
            name(str): The name of the parameter
            value(Any, optional): The value of the parameter. If empty, the parameter must be in local vars
            **kwargs: Other arguments. See DSSParameter help.
        """
        if self.local_vars:
            value = value or self._get_local_var(name)
        self.config[name] = DSSParameter(name=name, value=value, **kwargs)

    def get_param(self, name: AnyStr) -> DSSParameter:
        """Returns the DSSParameter of given name

        Args:
            name(str): Name of object to return

        Returns:
            DSSParameter: Parameter of given name
        """
        return self.config.get(name)

    def _get_local_var(self, var_name: AnyStr) -> Any:
        """Returns the value of the local variable related to var_name.

        Args:
            var_name(str): The variable to fetch from local_vars. It will be prefixed by the attribute "local_prefix"

        Returns:
            Any: The value matching the given name
        """
        return self.local_vars.get('{}{}'.format(self.local_prefix, var_name), None)

    def __delitem__(self, item):
        del self.config[item]

    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, key, value):
        self[key] = value

    def __getitem__(self, item):
        if item in self.config:
            return self.config.get(item).value
        else:
            raise KeyError(item)

    def __setitem__(self, key, value):
        self.add_param(name=key, value=value)

    def __iter__(self):
        return iter(self.config)

    def __len__(self):
        return len(self.config)

    def __repr__(self):
        return self.config.__repr__()

    def __str__(self):
        return self.config.__str__()
