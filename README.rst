================
SpectrumAnalysis
================

Spectrum Analysis tool for THz Time-domain data analysis 

* Free software: GNU General Public License v3

Installation
--------
Use the following command to install this python package ：
$ pip install －e .

Usage
--------
1. Compress the folder containing all your txt file data into a zip file.

2. Put the zip file into the same directory of your jupyter notebook

3. In the notebook, simply import SpectrumAnalysis to analyze your data. You may refer to the notebooks in demo/ for examples.

Features
--------

1. Easy to use by using jupyter notebook interface

2. Require no data preprocessing before usage

3. Can automatically identify data filenames , no need to hard code in the data filenames

4. Generate plots with single line of command

5. Store raw data and statistical analysis data automatically into fits files.

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
