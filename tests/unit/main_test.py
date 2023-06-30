import sys

sys.path.append("/home/parinayok/nutrition5k/src")

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
    main()


@mock.patch(
    "argparse.ArgumentParser.parse_args",
    return_value=argparse.Namespace(cfg="tests/config/test3.yaml"),
)
def test3(mock_args):
    main()
