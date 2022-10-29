from setuptools import setup

def _requires_from_file(filename):
    ls = open(filename).read().splitlines()
    return list(filter(lambda x : not (len(x) == 0 or x.startswith('#')), ls))

setup(
    name = "yolov7",
    install_requires = _requires_from_file('requirements.txt')
)