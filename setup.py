import os
import re
import sys
import sysconfig
import platform
import subprocess
import multiprocessing

from packaging.version import Version
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.install_lib import install_lib


# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}


def get_cmake_bool_flag(name, default_value=None):
    """
    Get cmake boolean flag from environment variables.
    `default_value` input can be either None or a bool
    """
    true_ = ("on", "true", "1", "t")  # Add more entries if you want...
    false_ = ("off", "false", "0", "f")  # Add more entries if you want...
    value = os.getenv(name, None)
    if value is None:
        if default_value is None:
            raise ValueError(f"Variable `{name}` not set!")
        else:
            value = str(default_value)
    if value.lower() not in true_ + false_:
        raise ValueError(f"Invalid value `{value}` for variable `{name}`")
    return value.lower() in true_


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class InstallCMakeLibs(install_lib):
    """
    Get the libraries from the parent distribution, use those as the outfiles
    Skip building anything; everything is already built, forward libraries to
    the installation step
    """

    def run(self):
        """
        Copy libraries from the bin directory and place them as appropriate
        """

        self.announce("Moving library files", level=3)

        # We have already built the libraries in the previous build_ext step
        self.skip_build = True

        # Folder where the `fastddm` package has been placed by cmake.
        # It is used by self.install
        self.build_dir = self.distribution.lib_dir
        self.outfiles = self.install()

        # I have copied this bit from the parent class
        if self.outfiles is not None:
            # always compile, in case we have any extension stubs to deal with
            self.byte_compile(self.outfiles)

    def get_outputs(self):
        """
        Overrides the parent class' method.
        Returns a list of the files copied over by the `run` method
        """
        return self.outfiles


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        self.cmake_version = Version(
            re.search(r"version\s*([\d.]+)", out.decode()).group(1)
        )

        if platform.system() == "Windows":
            if self.cmake_version < Version("3.1.0"):
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        self.announce("Preparing the build environment", level=3)

        cmake_args = []

        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]
        native_generator_args = ["--"]

        if platform.system() == "windows":
            if sys.maxsize > 2**32:
                cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]
            native_generator_args += ["/m"]
        else:
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]

        if sys.platform.startswith("darwin"):
            # Cross-compile support for macOS - respect ARCHFLAGS if set
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += ["-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]

        # Set optional CMake flags
        # C++
        if get_cmake_bool_flag("ENABLE_CPP", False):
            cmake_args += ["-DENABLE_CPP=ON"]
        # CUDA
        if get_cmake_bool_flag("ENABLE_CUDA", False):
            if platform.system() in ["Windows", "Linux"]:
                cmake_args += ["-DENABLE_CUDA=ON"]
            else:
                raise RuntimeError("Cannot build with CUDA on MacOS.")
            # if ENABLE_CPP is OFF, set it ON
            if not get_cmake_bool_flag("ENABLE_CPP", False):
                cmake_args += ["-DENABLE_CPP=ON"]
        # single precision
        if get_cmake_bool_flag("SINGLE_PRECISION", False):
            cmake_args += ["-DSINGLE_PRECISION=ON"]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            try:
                cpu_cores = int(os.getenv("SLURM_NTASKS"))
            except:
                cpu_cores = int(multiprocessing.cpu_count() / 2)

            if self.cmake_version < Version("3.14.0"):
                native_generator_args += [f"-j{cpu_cores}"]
            else:
                build_args += [f"-j {cpu_cores}"]

        build_args += native_generator_args

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        self.distribution.lib_dir = os.path.join(self.build_temp, "src/python")

        self.announce("Configuring cmake project", level=3)
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp
        )

        self.announce("Building the library", level=3)
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )

        self.announce("Compilation done", level=3)


setup(
    name="fastddm",
    use_scm_version={
        "fallback_version": "0.3.12",
    },
    packages=find_packages(),
    setup_requires=[
        "setuptools_scm",
    ],
    url="https://github.com/somexlab/fastddm",
    description=(
        "A Python/C++ library for the analysis of "
        "Differential Dynamic Microscopy experiments"
    ),
    long_description=open("./README.md", "r").read(),
    long_description_content_type="text/markdown",
    license="GNU GPL 3.0",
    ext_modules=[CMakeExtension("fastddm")],
    # add custom build_ext command
    cmdclass={
        "build_ext": CMakeBuild,
        "install_lib": InstallCMakeLibs,
    },
    zip_safe=False,
)
