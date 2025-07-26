from setuptools import find_packages
from setuptools import setup

setup(
    name='final_pipeline',
    version='0.0.1',
    packages=find_packages(
        include=('final_pipeline', 'final_pipeline.*')),
)
