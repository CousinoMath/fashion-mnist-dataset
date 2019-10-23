from setuptools import setup, find_packages

setup(name="fashion-mnist",
    version="0.0",
    install_requires=('docutils', 'numpy', 'tensorflow'),
    package_data={
        '': ['*.txt', '*.rst', '*.gz']
    }
)