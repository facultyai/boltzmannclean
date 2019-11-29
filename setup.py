import os
from setuptools import setup


def read_long_description():
    with open(os.path.join(os.path.dirname(__file__), "README.rst")) as fp:
        return fp.read()


setup(
    name="boltzmannclean",
    version="0.1.2",
    url="https://github.com/facultyai/boltzmannclean",
    author="Faculty",
    author_email="opensource@faculty.ai",
    description="Fill missing values in DataFrames with Restricted Boltzmann Machines",
    license="Apache 2.0",
    long_description=read_long_description(),
    py_modules=["boltzmannclean"],
    install_requires=["pandas", "numpy", "scipy", "scikit-learn"],
)
