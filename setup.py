from setuptools import setup, find_packages

setup(
    name="bandit_changepoint",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'pytest'
    ]
)