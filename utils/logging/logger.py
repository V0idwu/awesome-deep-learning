import logging

import colorlog


class ConsoleLogger:

    LOGLEVEL_2_COLOR = {
        "INFO": "green",
        "DEBUG": "cyan",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold_red",
    }

    def __init__(
        self, log_name: str = "root", log_level: int = logging.DEBUG, formatter_type: str = "simple", use_colorlog: bool = True
    ):
        self.__name = log_name
        self.__level = log_level
        self.__use_colorlog = use_colorlog
        self.init_logger(formatter_type)

    def init_logger(self, formatter_type):
        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(self.__level)
        if formatter_type == "detail":
            formatter = logging.Formatter(
                fmt=f"[%(asctime)s] [{self.__name}] [%(filename)s %(threadName)s -> %(funcName)s line:%(lineno)d] [%(levelname)s]: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

            if self.__use_colorlog:
                formatter = colorlog.ColoredFormatter(
                    fmt=f"%(log_color)s[%(asctime)s] [{self.__name}] [%(filename)s %(threadName)s -> %(funcName)s line:%(lineno)d] [%(levelname)s]: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    log_colors=self.LOGLEVEL_2_COLOR,
                )

        elif formatter_type == "simple":
            formatter = logging.Formatter(
                fmt=f"[%(asctime)s] [%(levelname)s]: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            if self.__use_colorlog:
                formatter = colorlog.ColoredFormatter(
                    fmt=f"%(log_color)s[%(asctime)s] [%(levelname)s]: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    log_colors=self.LOGLEVEL_2_COLOR,
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


# class CustomFilter(logging.Filter):
#     def filter(self, record):
#         return "loss" in record.getMessage()
class MyLogger:
    """A custom logging class that initializes based on the log parameters."""

    _NAME2LEVEL = {
        "CRITICAL": 50,
        "FATAL": 50,
        "ERROR": 40,
        "WARN": 30,
        "WARNING": 30,
        "INFO": 20,
        "DEBUG": 10,
        "NOTSET": 0,
    }

    def __init__(self, name: str = "root", log_params: dict = {}):
        self.__name = name
        self.__handlers = []
        for log_name, log_param in log_params.items():
            if log_param["log_type"] == "console":
                handler = self.init_console_handler(log_name, log_param)
            elif log_param["log_type"] == "file":
                handler = self.init_file_handler(log_name, log_param)
            else:
                raise ValueError(f"log_type: {log_param['log_type']} is not supported")
            self.__handlers.append(handler)

    def _log_formatter(self, log_name, log_formatter_type):
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
        return formatter

    def init_console_handler(self, log_name, log_param):
        console_handler = logging.StreamHandler()
        log_level = self._NAME2LEVEL[log_param.get("log_level", logging.DEBUG)]
        console_handler.setLevel(log_level)
        log_formatter = self._log_formatter(log_name, log_param.get("log_formatter_type", "simple"))
        console_handler.setFormatter(log_formatter)
        # log_filter_value = log_param.get("log_filter", None)
        # if log_filter_value is not None:
        #     filter = CustomFilter(log_filter_value)
        #     console_handler.addFilter(filter)
        return console_handler

    def init_file_handler(self, log_name, log_param):
        log_path = log_param.get("log_path", f"{log_name}.log")
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        log_level = self._NAME2LEVEL[log_param.get("log_level", logging.DEBUG)]
        file_handler.setLevel(log_level)
        log_formatter = self._log_formatter(log_name, log_param.get("log_formatter_type", "simple"))
        file_handler.setFormatter(log_formatter)
        # log_filter_value = log_param.get("log_filter", None)
        # if log_filter_value is not None:
        #     filter = CustomFilter(log_filter_value)
        #     file_handler.addFilter(filter)
        return file_handler

    def get_logger(self):
        logger = logging.getLogger(self.__name)
        logger.setLevel(logging.DEBUG)
        for handler in self.__handlers:
            logger.addHandler(handler)
        return logger


if __name__ == "__main__":

    # Test ConsoleLogger
    logger = ConsoleLogger(log_name=__name__).get_logger()
    logger.debug("hello world")
    logger.info("hello world")
    logger.error("hello world")
    logger.warning("hello world")
    logger.critical("hello world")

    # import logging.config

    # logging.config.fileConfig("utils\logging\logger.conf")
    # root_logger = logging.getLogger("root_log")
    # loss_logger = logging.getLogger("loss_log")
    # acc_logger = logging.getLogger("accurate_log")

    # Test MyLogger
    params = {
        "train.acc": {
            "log_type": "file",
            "log_path": "train.log",
            "log_level": "INFO",
            "log_formatter_type": "standard",
            # "log_filter": "train.acc",
        },
        "train.loss": {
            "log_type": "console",
            "log_level": "DEBUG",
            "log_formatter_type": "standard",
        },
    }

    import time

    logger = MyLogger(log_params=params).get_logger()
    logger.info(time.time())
