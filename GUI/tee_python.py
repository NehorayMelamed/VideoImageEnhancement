import sys

class Tee:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

if __name__ == "__main__":
    log_file = sys.argv[1]
    sys.stdout = Tee(log_file)
    sys.stderr = Tee(log_file)

    # Execute the script passed as argument
    exec(open(sys.argv[2]).read())
