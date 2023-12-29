import logging
from re import M

import colorlog


# NOTE: Print log to console
class ConsoleLogger:
    def __init__(self, log_name: str = "root", log_level: int = logging.DEBUG, formatter_type="simple"):
        self.__name = log_name
        self.__level = log_level
        self.init_logger(formatter_type)

    def init_logger(self, formatter_type):
        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(self.__level)
        if formatter_type == "standard":
            formatter = logging.Formatter(
                fmt=f"[%(asctime)s] [{self.__name}] [%(filename)s %(threadName)s -> %(funcName)s line:%(lineno)d] [%(levelname)s]: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        elif formatter_type == "simple":
            formatter = logging.Formatter(
                fmt=f"[%(asctime)s] [%(levelname)s]: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        self.console_handler.setFormatter(formatter)

    def get_logger(self):
        self.__logger = logging.getLogger(self.__name)
        self.__logger.setLevel(logging.DEBUG)
        self.__logger.addHandler(self.console_handler)
        return self.__logger

    @property
    def level(self):
        return self.__level

    @property
    def name(self):
        return self.__name


class CConsoleLogger:
    LOG_COLOR_MAP = {
        "INFO": "green",
        "DEBUG": "cyan",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold_red",
    }

    def __init__(self, log_name: str = "default", log_level: int = logging.DEBUG, formatter_type="simple"):
        self.__name = log_name
        self.__level = log_level
        self.init_logger(formatter_type)

    def init_logger(self, formatter_type):
        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(self.__level)
        if formatter_type == "standard":
            formatter = colorlog.ColoredFormatter(
                fmt=f"%(log_color)s[%(asctime)s] [{self.__name}] [%(filename)s %(threadName)s -> %(funcName)s line:%(lineno)d] [%(levelname)s]: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                log_colors=self.LOG_COLOR_MAP,
            )
        elif formatter_type == "simple":
            formatter = colorlog.ColoredFormatter(
                fmt=f"%(log_color)s[%(asctime)s] [%(levelname)s]: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                log_colors=self.LOG_COLOR_MAP,
            )
        self.console_handler.setFormatter(formatter)

    def get_logger(self):
        self.__logger = logging.getLogger(self.__name)
        self.__logger.setLevel(logging.DEBUG)
        self.__logger.addHandler(self.console_handler)
        return self.__logger

    @property
    def level(self):
        return self.__level

    @property
    def name(self):
        return self.__name


# NOTE: Print log to file
class FileLogger:
    def __init__(self, log_name: str = "default", log_level: int = logging.DEBUG) -> None:
        self.__name = log_name
        self.__level = log_level
        self.init_file_handler()

    def init_file_handler(self) -> None:
        self.__file_handler = logging.FileHandler(f"{self.__name}.log", encoding="utf-8")
        self.__file_handler.setLevel(self.__level)
        self.__file_handler.setFormatter(
            logging.Formatter(
                fmt=f"[%(asctime)s] [{self.__name}] [%(filename)s %(threadName)s -> %(funcName)s line:%(lineno)d] [%(levelname)s]: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    def get_logger(self) -> logging.Logger:
        self.__logger = logging.getLogger(self.__name)
        self.__logger.setLevel(self.__level)
        self.__logger.addHandler(self.__file_handler)
        return self.__logger


class ConfLogger:
    def __init__(self, conf_path: str = None) -> None:
        if conf_path is not None:
            logging.config.fileConfig(conf_path)
        else:
            raise ValueError("conf_path is None")

    def get_logger(self):
        root_logger = logging.getLogger("root_log")
        loss_logger = logging.getLogger("loss_log")
        acc_logger = logging.getLogger("accurate_log")
        return root_logger, loss_logger, acc_logger


# NOTE: Print log to console and file
class MyLogger:
    _nameToLevel = {"CRITICAL": 50, "FATAL": 50, "ERROR": 40, "WARN": 30, "WARNING": 30, "INFO": 20, "DEBUG": 10, "NOTSET": 0}

    def __init__(self, name: str = "root", log_params: dict = {}):
        self.__name = name
        self.__handlers = []
        self.__filters = []
        for log_name, log_param in log_params.items():
            if log_param["log_type"] == "console":
                handler = self.init_console_handler(log_name, log_param)
            elif log_param["log_type"] == "file":
                handler = self.init_file_handler(log_name, log_param)
            else:
                raise ValueError(f"log_type: {log_param['log_type']} is not supported")
            self.__handlers.append(handler)

    def init_console_handler(self, log_name, log_param):
        console_handler = logging.StreamHandler()
        log_level = self._nameToLevel[log_param.get("log_level", logging.DEBUG)]
        console_handler.setLevel(log_level)
        log_formatter_type = log_param.get("log_formatter_type", "simple")
        if log_formatter_type == "standard":
            formatter = logging.Formatter(
                fmt=f"[%(asctime)s] [{log_name}] [%(filename)s %(threadName)s -> %(funcName)s line:%(lineno)d] [%(levelname)s]: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        elif log_formatter_type == "simple":
            formatter = logging.Formatter(
                fmt=f"[%(asctime)s] [%(levelname)s]: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        console_handler.setFormatter(formatter)
        log_filter = log_param.get("log_filter", None)
        if log_filter is not None:
            filter = logging.Filter(log_filter)
            console_handler.addFilter(filter)
        return console_handler

    def init_file_handler(self, log_name, log_param):
        log_path = log_param.get("log_path", f"{log_name}.log")
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        log_level = self._nameToLevel[log_param.get("log_level", logging.DEBUG)]
        file_handler.setLevel(log_level)
        log_formatter_type = log_param.get("log_formatter_type", "simple")
        if log_formatter_type == "standard":
            formatter = logging.Formatter(
                fmt=f"[%(asctime)s] [{log_name}] [%(filename)s %(threadName)s -> %(funcName)s line:%(lineno)d] [%(levelname)s]: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        elif log_formatter_type == "simple":
            formatter = logging.Formatter(
                fmt=f"[%(asctime)s] [%(levelname)s]: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        file_handler.setFormatter(formatter)

        log_filter = log_param.get("log_filter", None)
        if log_filter is not None:
            filter = logging.Filter(log_filter)
            file_handler.addFilter(filter)
        return file_handler

    def get_logger(self):
        logger = logging.getLogger(self.__name)
        logger.setLevel(logging.DEBUG)
        for handler in self.__handlers:
            logger.addHandler(handler)
        for filter in self.__filters:
            logger.addFilter(filter)
        return logger


if __name__ == "__main__":
    # logger = ConsoleLogger("train.loss.log").get_logger()
    # logger.info("hello world")

    # logger = CConsoleLogger(__name__).get_logger()
    # logger.debug("hello world")
    # logger.info("hello world")
    # logger.error("hello world")
    # logger.warning("hello world")
    # logger.critical("hello world")

    # import logging.config

    # logging.config.fileConfig("utils\logging\logger.conf")
    # root_logger = logging.getLogger("root_log")
    # loss_logger = logging.getLogger("loss_log")
    # acc_logger = logging.getLogger("accurate_log")

    params = {
        "train.acc": {
            "log_type": "file",
            "log_path": "train.log",
            "log_level": "INFO",
            "log_formatter_type": "simple",
            "log_filter": "train.acc",
        },
        "train.loss": {
            "log_type": "console",
            "log_level": "DEBUG",
            "log_formatter_type": "standard",
        },
    }

    import time

    losslogger = MyLogger(name="learn", log_params=params).get_logger()
    losslogger.info(time.time())

    # acclogger = MyLogger(logger_config).get_logger("train.acc")
    # acclogger.info("hello world111")
