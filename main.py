import argparse
import logging

from errors import Error
from eval_cmd import register_eval_cmd, handle_eval_cmd
from plot_cmd import handle_plot_cmd, register_plot_args
from ploti_cmd import register_ploti_args, handle_ploti_cmd
from recognize_cmd import register_recognize_args, handle_recognize_cmd
from train_cmd import handle_train_cmd, register_train_args

LOG = logging.getLogger(__name__)


def main():
    try:
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="cmd")

        eval_cmd = subparsers.add_parser("eval", help="Runs neural network evaluation")
        register_eval_cmd(eval_cmd)

        plot_cmd = subparsers.add_parser("plot", help="Plot dataset sample")
        register_plot_args(plot_cmd)

        ploti_cmd = subparsers.add_parser("ploti", help="Plot dataset samples in interactive mode")
        register_ploti_args(ploti_cmd)

        train_cmd = subparsers.add_parser("train", help="Runs neural network training and then evaluation")
        register_train_args(train_cmd)

        recognize_cmd = subparsers.add_parser("recognize", help="Runs text recognition")
        register_recognize_args(recognize_cmd)

        args = parser.parse_args()

        if args.cmd == "eval":
            handle_eval_cmd(args)
        elif args.cmd == "train":
            handle_train_cmd(args)
        elif args.cmd == "recognize":
            handle_recognize_cmd(args)
        elif args.cmd == "plot":
            handle_plot_cmd(args)
        elif args.cmd == "ploti":
            handle_ploti_cmd(args)

        return 0

    except Error as e:
        LOG.error(f"Error: {e.message}")
        return 1
