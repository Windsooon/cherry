"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='cherry',
    version='0.4.0',
    description='classify data with native bayes',
    long_description=long_description,
    url='https://github.com/Sunkist-Cherry/cherry',
    author='Windson Yang',
    author_email='wiwindson@outlook.com',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='data classify content filter',
    install_requires=[
        'jieba>=0.39',
        'numpy>=1.13.3',
        'terminaltables>=3.1.0',
        'nltk>=3.2.5',
        'matplotlib>=2.2.2'
        ],
    packages=['cherry'],
    package_data={
        'cherry': [
            'data/Chinese/cache/*', 'data/Chinese/*.dat',
            'data/English/cache/*', 'data/English/*.dat']
    },
)
