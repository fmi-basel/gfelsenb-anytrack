import os
import setuptools
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="anytrack",
    version="1.0.0",
    author="Dennis Goldschmidt",
    author_email="dennis.goldschmidt@fmi.ch",
    date="2024.08.07",
    description=("Library for logging, processing, visualization and sharing of image-based tracking data of fruit flies written in Python."),
    license="GPLv3",
    keywords=['tracking', 'data analysis', 'fly'],
    #url="https://pypi.python.org/pypi/anytrack",
    packages=setuptools.find_packages(),
    python_requires='>=3.12.4',
    long_description=read('README.md'),
    classifiers=[],
    platforms=['MacOS Sonoma 14.5 (23F79)'],
    install_requires=['setuptools','opencv-python','numpy'],
    #entry_points={
        #'console_scripts': [
        #    'anytrack = anytrack.main:main',
        #    'anytrack-compress = anytrack.compressor:main',
        #    'anytrack-label = anytrack.annotator:main',
        #],
        #'gui_scripts': [
        #    'anytrack-gui = anytrack.app:main',
        #],
    #},
)
