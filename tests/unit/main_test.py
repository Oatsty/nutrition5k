import sys

import torch

sys.path.insert(0, "/".join([sys.path[0], "src"]))

import argparse
from unittest import mock

from src.main import main


@mock.patch(
    "argparse.ArgumentParser.parse_args",
    return_value=argparse.Namespace(cfg="tests/config/test1.yaml"),
)
def test1(mock_args):
    main()


@mock.patch(
    "argparse.ArgumentParser.parse_args",
    return_value=argparse.Namespace(cfg="tests/config/test2.yaml"),
)
def test2(mock_args):
    if not torch.cuda.is_available():
        return
    main()


@mock.patch(
    "argparse.ArgumentParser.parse_args",
    return_value=argparse.Namespace(cfg="tests/config/test3.yaml"),
)
def test3(mock_args):
    main()
