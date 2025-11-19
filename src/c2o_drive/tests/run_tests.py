#!/usr/bin/env python3
"""Unified test runner for C2O-Drive."""

import sys
import argparse
from pathlib import Path
import pytest


def main():
    """Run tests based on command line arguments."""
    parser = argparse.ArgumentParser(description="Run C2O-Drive tests")
    parser.add_argument(
        "--type",
        choices=["all", "unit", "integration", "functional"],
        default="all",
        help="Type of tests to run",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report",
    )
    parser.add_argument(
        "--markers", "-m",
        help="Run tests matching given mark expression",
    )
    parser.add_argument(
        "--file", "-f",
        help="Run specific test file",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip slow tests",
    )
    parser.add_argument(
        "--no-carla",
        action="store_true",
        help="Skip tests requiring CARLA",
    )

    args = parser.parse_args()

    # Build pytest arguments
    pytest_args = []

    # Add test directory based on type
    test_dir = Path(__file__).parent
    if args.file:
        pytest_args.append(args.file)
    elif args.type == "all":
        pytest_args.append(str(test_dir))
    else:
        pytest_args.append(str(test_dir / args.type))

    # Add verbosity
    if args.verbose:
        pytest_args.append("-v")

    # Add coverage
    if args.coverage:
        pytest_args.extend([
            "--cov=c2o_drive",
            "--cov-report=html",
            "--cov-report=term",
        ])

    # Add markers
    markers = []
    if args.markers:
        markers.append(args.markers)
    if args.quick:
        markers.append("not slow")
    if args.no_carla:
        markers.append("not carla")

    if markers:
        pytest_args.extend(["-m", " and ".join(markers)])

    # Add color output
    pytest_args.append("--color=yes")

    # Run pytest
    print(f"Running tests with args: {' '.join(pytest_args)}")
    return pytest.main(pytest_args)


if __name__ == "__main__":
    sys.exit(main())