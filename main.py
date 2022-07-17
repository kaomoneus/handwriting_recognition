import argparse
import logging

from commands import mk_lines_cmd, synthetic_cmd
from errors import Error
from commands.eval_cmd import register_eval_cmd, handle_eval_cmd
from commands.plot_cmd import handle_plot_cmd, register_plot_args
from commands.ploti_cmd import register_ploti_args, handle_ploti_cmd
from commands.recognize_cmd import register_recognize_args, handle_recognize_cmd
from commands.train_cmd import handle_train_cmd, register_train_args

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

        mk_lines_command = subparsers.add_parser(
            "mk-lines",
            help="Renders lines, taking into account blacklisted words"
        )
        mk_lines_cmd.register(mk_lines_command)

        synthetic_command = subparsers.add_parser(
            "synthetic",
            help="Generates synthetic dataset"
        )
        synthetic_cmd.register(synthetic_command)

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
        elif args.cmd == "mk-lines":
            mk_lines_cmd.handle(args)
        elif args.cmd == "synthetic":
            synthetic_cmd.handle(args)

        return 0

    except Error as e:
        LOG.error(f"Error: {e.message}")
        return 1
