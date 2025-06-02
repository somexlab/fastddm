"""
Simple script to handle the installation of the package and its dependencies.
Removes the need to use lengthy command line options.
"""

import argparse
import subprocess
import sys
import logging
from dataclasses import dataclass

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("install.log", mode="w")
    ],
)

logger = logging.getLogger(__name__)
logger.info("Starting installation...")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Install the fastddm package with custom build and dependency options."
    )
    parser.add_argument(
        "--cpp",
        action="store_true",
        help="Enable C++ backend (sets ENABLE_CPP=ON)."
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Enable CUDA GPU backend (sets ENABLE_CUDA=ON)."
    )
    parser.add_argument(
        "--single-prec",
        action="store_true",
        help="Enable single precision (sets SINGLE_PRECISION=ON)."
    )
    parser.add_argument(
        "--uv",
        action="store_true",
        help="Use 'uv' as the installer instead of pip."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output."
    )
    parser.add_argument(
        "-e", "--editable",
        action="store_true",
        help="Install in editable mode."
    )
    parser.add_argument(
        "--no-cache-dir",
        action="store_true",
        help="Disable cache during installation."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Install test dependencies."
    )
    parser.add_argument(
        "--doc",
        action="store_true",
        help="Install documentation dependencies."
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Install development dependencies."
    )
    return parser.parse_args()

@dataclass
class CMakeConfigSettings:
    enable_cpp: bool = False
    enable_cuda: bool = False
    single_precision: bool = False

    def as_config_args(self):
        args = []
        if self.enable_cpp:
            logging.info("C++ backend enabled.")
            args.append("--config-settings=cmake.define.ENABLE_CPP=ON")
        if self.enable_cuda:
            logging.info("CUDA GPU backend enabled.")
            args.append("--config-settings=cmake.define.ENABLE_CUDA=ON")
        if self.single_precision:
            logging.info("Single precision enabled.")
            args.append("--config-settings=cmake.define.SINGLE_PRECISION=ON")
        return args

class Installer:
    def __init__(self, args, cmake_settings: CMakeConfigSettings):
        self.args = args
        self.cmake_settings = cmake_settings

    def build_extras(self):
        extras = []
        if self.args.test:
            logging.info("Test dependencies will be installed.")
            extras.append("test")
        if self.args.doc:
            logging.info("Documentation dependencies will be installed.")
            extras.append("doc")
        if self.args.dev:
            logging.info("Development dependencies will be installed.")
            extras.append("dev")
        return extras

    def build_target(self, extras):
        if extras:
            return f".[{','.join(extras)}]"
        return "."

    def build_command(self):
        if self.args.uv:
            logging.info("Using 'uv' as the installer.")
            installer = ["uv", "pip", "install"]
        else:
            logging.info("Using 'pip' as the installer.")
            installer = [sys.executable, "-m", "pip", "install"]

        extras = self.build_extras()
        target = self.build_target(extras)

        cmd = installer.copy()
        if self.args.editable:
            logging.info("Installing in editable mode.")
            cmd.append("-e")
        cmd.append(target)

        if self.args.no_cache_dir:
            logging.info("Disabling cache during installation.")
            cmd.append("--no-cache-dir")
        if self.args.verbose:
            logging.info("Verbose output enabled.")
            cmd.append("-v")

        # Add config settings from the dataclass
        cmd.extend(self.cmake_settings.as_config_args())

        return cmd

    def run(self):
        cmd = self.build_command()
        logging.debug(f"Running command: {' '.join(cmd)}")
        ret = subprocess.run(cmd, check=True)
        logger.debug(f"Command returned with code: {ret.returncode}")

def main() -> None:
    # Step 1: Parse command line arguments
    args = parse_args()
    # Step 2: Build config-settings arguments for CMake options using dataclass
    cmake_settings = CMakeConfigSettings(
        enable_cpp=args.cpp,
        enable_cuda=args.gpu,
        single_precision=args.single_prec,
    )
    # Step 3: Build and run install command
    installer = Installer(args, cmake_settings)
    installer.run()

if __name__ == "__main__":
    main()
