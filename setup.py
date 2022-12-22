import glob
from os.path import basename
from os.path import splitext
from setuptools import setup

def _requires_from_file(filename):
    with open(filename) as f:
        ls = f.read().splitlines()
    return list(filter(lambda x : not (len(x) == 0 or x.startswith('#')), ls))

setup(
    name = "yolov7",
    packages=[""],
    package_dir={"": "yolov7"},
    py_modules=[splitext(basename(path))[0] for path in glob.glob('*.py')],
    install_requires = _requires_from_file('requirements.txt')
)