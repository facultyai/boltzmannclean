from setuptools import setup
import os


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

# README = Path(__file__).parent / 'README.rst'


setup(
    name='sherlockml-boltzmannclean',
    version='0.1.2',
    url='https://sherlockml.com',
    author='ASI Data Science',
    author_email='engineering@asidatascience.com',
    description='Fills missing values in a pandas DataFrame using a Restricted'
                ' Boltzmann Machine.',
    license='Apache 2.0',
    long_description=read('README.rst'),
    py_modules=['boltzmannclean'],
    install_requires=[
        'pandas',
        'numpy',
        'scipy',
        'scikit-learn'
    ]
)
