from setuptools import setup

import sys

if sys.version_info < (3, 8):
    sys.exit("Error, Python < 3.8 is not supported.")

setup(
    name="fastddm",
    use_scm_version={
        "root": "${CMAKE_SOURCE_DIR}",
        "fallback_version": "0.3.12",
    },
    setup_requires=["setuptools_scm"],
    packages=["fastddm"],
    package_dir={"fastddm": "${FASTDDM_OUTPUT_DIR}"},
    package_data={"fastddm": ["_core.*", "_core_cuda.*"]},
)
