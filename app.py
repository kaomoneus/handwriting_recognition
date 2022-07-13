import sys

from utils.log_utils import init_logging

if __name__ == "__main__":
    init_logging()
    from main import main
    res = main()
    sys.exit(res)
