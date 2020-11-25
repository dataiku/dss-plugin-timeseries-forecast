import logging
import copy

MESSAGE_TEMPLATE = "{} - {}"


class SafeLogger(object):
    def __init__(self, name, forbiden_keys=None):
        self.name = name
        self.logger = logging.getLogger(self.name)
        # logging.basicConfig(
        #     level=logging.INFO,
        #     format='{} %(levelname)s - %(message)s'.format(self.name)
        # )
        self.forbiden_keys = forbiden_keys

    def info(self, message):
        self.logger.info(MESSAGE_TEMPLATE.format(self.name, message))

    def debug(self, message):
        self.logger.debug(MESSAGE_TEMPLATE.format(self.name, message))

    def warning(self, message):
        self.logger.warning(MESSAGE_TEMPLATE.format(self.name, message))

    def error(self, message):
        self.logger.error(MESSAGE_TEMPLATE.format(self.name, message))

    def filter_secrets(self, dictionary):
        ret = copy.deepcopy(dictionary)
        ret = self.dig_secrets(ret)
        return ret

    def dig_secrets(self, dictionary):
        for key in dictionary:
            if isinstance(dictionary[key], dict):
                dictionary[key] = self.filter_secrets(dictionary[key])
            if key in self.forbiden_keys:
                dictionary[key] = hash(dictionary[key])
        return dictionary


def hash(data):
    data_type = type(data).__name__
    if data_type in ["str", "dict", "list", "unicode"]:
        data_len = len(data)
    else:
        data_len = 0
    return "HASHED_SECRET:{}:{}".format(data_type, data_len)
