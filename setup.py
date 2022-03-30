from setuptools import setup, find_packages

setup(
    name='deep_hvac',
    version='0.0.0',
    description='HVAC control using deep RL',
    author='Daniel Shen & Cale Gregory',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'gym',
        'openpyxl',
        'pandas',
        'pysolar',
        'tensorboard',
        'torch'
        ],
    extras_require={
        'test': ['pytest', 'mock'],
    },
    project_urls={},
)
