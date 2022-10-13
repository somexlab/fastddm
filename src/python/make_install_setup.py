from setuptools import setup

import sys
if sys.version_info < (3,6):
    sys.exit("Error, Python < 3.6 is not supported.")

setup(
    name = "dfmtoolbox",
    use_scm_version = {
        "root": "${CMAKE_SOURCE_DIR}",
        "fallback_version": "0.1.0",
    },
    setup_requires = ["setuptools_scm"],
    packages = ["dfmtoolbox"],
    package_dir = {
        "dfmtoolbox": "${DFMTOOLBOX_OUTPUT_DIR}"
    },
    package_data = {
        "dfmtoolbox": ['core.so']
    },
)
