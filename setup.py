from setuptools import setup, find_packages

setup(
    name="bandit_changepoint",
    version="0.1",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        'numpy>=1.24.0',
        'matplotlib>=3.7.0',
        'pandas>=1.5.0',
        'tqdm>=4.0.0',
    ],
    extras_require={
        'test': ['pytest>=7.0.0'],
    },
)
