"""
Simple script to handle the installation of the package and its dependencies.
Removes the need to use lengthy command line options.
"""

import argparse
import glob
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import logging
from dataclasses import dataclass, field


def run_command(
    cmd: list[str],
    cwd: Path | None = None,
    check: bool = True,
    msg: str | None = None,
):
    logger = logging.getLogger(__name__)
    display_cmd = ' '.join(shlex.quote(arg) for arg in cmd)
    if msg is not None:
        logger.debug(msg)
    else:
        logger.debug("Executing command: %s", display_cmd)
    
    try:
        ret = subprocess.run(cmd, cwd=cwd, check=check)
        returncode = ret.returncode
        logger.debug(f"Command returned with code: {returncode}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with error: {e}")
        if e.stdout:
            logger.error(f"Standard output:\n{e.stdout.strip()}")
        if e.stderr:
            logger.error(f"Standard error:\n{e.stderr.strip()}")
        sys.exit(e.returncode)
    except FileNotFoundError:
        logger.error(f"Command not found: {cmd[0]}. Full command: {display_cmd}")
        sys.exit(-1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        logger.debug("Full command: %s", display_cmd)
        sys.exit(999)

def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the installer.

    Returns
    -------
    argparse.Namespace
        Parsed command line arguments.
    """
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
        "--pre-commit",
        action="store_true",
        help="Install pre-commit hooks",
    )
    parser.add_argument(
        "--extras",
        type=str,
        help="Comma-separated list of extra dependencies to install."
    )
    parser.add_argument(
        "--log-to-file",
        action="store_true",
        help="Log output to a file named 'install.log'."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the install command without executing it."
    )
    return parser.parse_args()

@dataclass(kw_only=True)
class CMakeConfigSettings:
    """
    Configuration settings for CMake build options.

    Attributes
    ----------
    enable_cpp : bool
        Indicates if the C++ backend is enabled. Default is False.
    enable_cuda : bool
        Indicates if the CUDA GPU backend is enabled. Default is False.
    single_precision : bool
        Indicates if single precision mode is enabled. Default is False.
    """
    enable_cpp: bool = False
    enable_cuda: bool = False
    single_precision: bool = False
    _logger: logging.Logger = field(init=False, default_factory=lambda: logging.getLogger(__name__))

    def as_config_args(self) -> list[str]:
        """Convert the dataclass fields to a list of config settings for CMake.
        
        Returns
        -------
        list[str]
            A list of CMake configuration arguments.
        """
        args = []
        if self.enable_cpp:
            self._logger.info("C++ backend enabled.")
            args.append("--config-settings=cmake.define.ENABLE_CPP=ON")
        if self.enable_cuda:
            self._logger.info("CUDA GPU backend enabled.")
            args.append("--config-settings=cmake.define.ENABLE_CUDA=ON")
        if self.single_precision:
            self._logger.info("Single precision enabled.")
            args.append("--config-settings=cmake.define.SINGLE_PRECISION=ON")
        return args

@dataclass(kw_only=True)
class Installer:
    """Installer class to handle the installation of the package.

    Attributes
    ----------
    args : argparse.Namespace
        Parsed command line arguments.
    cmake_settings : CMakeConfigSettings
        Configuration settings for CMake build options.
    """
    args: argparse.Namespace
    cmake_settings: CMakeConfigSettings
    _logger: logging.Logger = field(init=False, default_factory=lambda: logging.getLogger(__name__))

    def __post_init__(self):
        """Pre-installation checks."""
        self._logger.debug("Running pre-installation checks.")
        if self.args.uv:
            # Check if 'uv' is installed and executable
            if not self._check_command_is_available("uv"):
                self._logger.error(
                    "'uv' is not installed. "
                    "Please install it with 'pip install uv'."
                )
                sys.exit(1)
        if self.args.pre_commit:
            # Check if 'pre-commit' is installed and executable
            if not (self._check_command_is_available("pre-commit") or "dev" in self._extras()):
                self._logger.error(
                    "'pre-commit' is not installed. "
                    "Please install it with 'pip install pre-commit' or enable dev extra dependencies."
                )
                sys.exit(1)

    def _check_command_is_available(self, cmd: str) -> bool:
        """Check if given command 'cmd' is installed and executable.

        The command should implement the '--version' option to verify its availability.

        Parameters
        ----------
        cmd : str
            The command to check for availability (e.g., 'uv' or 'pre-commit').

        Returns
        -------
        bool
            True if command is installed and executable, False otherwise.
        """
        self._logger.debug(f"Checking if '{cmd}' is installed.")
        if shutil.which(cmd) is None:
            self._logger.debug(f"'{cmd}' command not found in PATH.")
            return False
        try:
            subprocess.run([cmd, "--version"], check=True, capture_output=True)
            self._logger.debug(f"'{cmd}' is installed and functional.")
            return True
        except subprocess.CalledProcessError:
            self._logger.warning(
                f"'{cmd}' was found in PATH, but '{cmd} --version' failed. "
                f"This might indicate a corrupted installation or an unexpected '{cmd}' version."
            )
            return False
    
    def hooks_install(self):
        pre_commit_files = glob.glob("**/.pre-commit-config.yaml", recursive=True)
        for file in pre_commit_files:
            self._logger.debug(f"Installing pre-commit hooks for {file}")
            directory = Path(file).parent.resolve()
            command = ["pre-commit", "install"]
            self._logger.debug(f"Running pre-commit install command: {' '.join(command)} in {directory}")
            ret = subprocess.run(command, cwd=directory)
            self._logger.debug(f"Pre-commit install returned with code: {ret.returncode}")

    def _extras(self) -> list[str]:
        """
        A list of extras to install.

        Returns
        -------
        list[str]
            A list of extras to install.
        """
        self._logger.debug("Building extras list from command line arguments.")
        if not self.args.extras or self.args.extras is None:
            self._logger.debug("No extras specified, returning empty list.")
            return []
        extras = [e.strip() for e in self.args.extras.split(",") if e.strip()]
        self._logger.info(f"Extras to install: {extras}")
        return extras

    def _target(self) -> str:
        """Build the target string for the installation command.

        Returns
        -------
        str
            The target string for the installation command.
        """
        self._logger.debug("Building target string for installation.")
        extras = self._extras()
        return f".[{','.join(extras)}]" if extras else "."

    def build_command(self) -> list[str]:
        """Build the command to run the installer.

        Returns
        -------
        list[str]
            The command to run the installer.
        """
        if self.args.uv:
            self._logger.info("Using 'uv' as the installer.")
            cmd = ["uv", "pip", "install"]
        else:
            self._logger.info("Using 'pip' as the installer.")
            cmd = [sys.executable, "-m", "pip", "install"]

        if self.args.editable:
            self._logger.info("Installing in editable mode.")
            cmd.append("-e")
        
        target = self._target()
        self._logger.debug(f"Target for installation: {target}")
        cmd.append(target)

        if self.args.no_cache_dir:
            self._logger.info("Disabling cache during installation.")
            cmd.append("--no-cache-dir")
        if self.args.verbose:
            self._logger.info("Verbose output enabled.")
            cmd.append("-v")

        # Add config settings from the dataclass
        cmd.extend(self.cmake_settings.as_config_args())

        return cmd

    def run(self) -> None:
        """Run the installation command."""
        cmd = self.build_command()
        if self.args.dry_run:
            dry_run_cmd = ' '.join(shlex.quote(arg) for arg in cmd)
            self._logger.info(f"Dry run: The following command would be executed:\n{dry_run_cmd}")
            return
        try:
            self._logger.debug("Running command: %s", ' '.join(shlex.quote(arg) for arg in cmd))
            project_dir = Path(__file__).parent.resolve()
            ret = subprocess.run(cmd, cwd=project_dir, check=True)
            self._logger.debug(f"Command returned with code: {ret.returncode}")
        except subprocess.CalledProcessError as e:
            self._logger.error(f"Installation failed with error: {e}")
            sys.exit(e.returncode)
        if self.args.pre_commit:
            self._logger.info("Installing pre-commit hooks.")
            self.hooks_install()

def main(args: argparse.Namespace) -> None:
    # Step 1: Build config-settings arguments for CMake options using dataclass
    cmake_settings = CMakeConfigSettings(
        enable_cpp=args.cpp,
        enable_cuda=args.gpu,
        single_precision=args.single_prec,
    )
    # Step 2: Build and run install command
    installer = Installer(args=args, cmake_settings=cmake_settings)
    installer.run()

if __name__ == "__main__":
    args = parse_args()
    # Set up logging
    logger_handlers = [logging.StreamHandler()]
    if args.log_to_file:
        logger_handlers.append(logging.FileHandler("install.log", mode="w"))
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=logger_handlers,
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting installation...")
    main(args=args)
