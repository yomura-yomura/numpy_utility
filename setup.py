from setuptools import setup, find_packages

setup(
    name='numpy_utility',
    version='1.7.2',
    description='',
    author='yomura',
    author_email='yomura@hoge.jp',
    url='https://github.com/yomura-yomura/numpy_utility',
    packages=find_packages(),
    install_requires=[
        "numpy",
        "more-itertools"
    ]
)
