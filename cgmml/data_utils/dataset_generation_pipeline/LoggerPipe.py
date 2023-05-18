class LoggerPipe:
    """Of this logger there can only be on instance (write mode)
    """
    def __init__(self, path_to_log: str):
        self.log_file = open(path_to_log, "w")

    def __del__(self):
        self.log_file.close()

    def write(self, message: str):
        print(message, file=self.log_file)

