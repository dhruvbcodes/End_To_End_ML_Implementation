# To build the ML model has a package so can we used whenever we want :)

from typing import List

from setuptools import setup, find_packages

HYPHEN_E_DOT = '-e .' # this is used to install the package in editable mode

# this function is used to read the requirements from the requirements.txt file
def get_requirements(filename:str) -> List[str]:
    requirements = []
    with open(filename, 'r') as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements

    

setup(
    name='ML_Project',
    version='0.0.1',
    author='Dhruv',
    author_email='dbaheti2003@gmail.com',
    packages=find_packages(),
    license='MIT',
    install_requires= get_requirements('requirements.txt')
)