import os
from setuptools import setup


def read_long_description():
    with open(os.path.join(os.path.dirname(__file__), "README.rst")) as fp:
        return fp.read()


setup(
    name="sherlockml-boltzmannclean",
    version="0.1.2",
    url="https://github.com/ASIDataScience/sherlockml-boltzmannclean",
    author="ASI Data Science",
    author_email="engineering@asidatascience.com",
    description="Fill missing values in DataFrames with Restricted Boltzmann Machines",
    license="Apache 2.0",
    long_description=read_long_description(),
    py_modules=["boltzmannclean"],
    install_requires=["pandas", "numpy", "scipy", "scikit-learn"],
)
