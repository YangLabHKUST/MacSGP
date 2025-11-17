from setuptools import Command, find_packages, setup

__lib_name__ = "MacSGP"
__lib_version__ = "0.1"
__description__ = "MacSGP is a scalable statistical and computational approach for MApping Cell-type-specific Spatial Gene Programs (SGPs) in spatial transcriptomic (ST) data."
__url__ = "https://github.com/YangLabHKUST/MacSGP"
__author__ = "Yeqin Zeng"
__author_email__ = "yzengbj@connect.ust.hk"

setup(
    name = __lib_name__,
    version = __lib_version__,
    description = __description__,
    url = __url__,
    author = __author__,
    author_email = __author_email__,
    zip_safe = False,
    include_package_data = True,
    # long_description=open("README.md").read(),
    # long_description_content_type="text/markdown", 
    license="MIT",
    license_files=("LICENSE",),
    packages=find_packages(),
    install_requires=[
        "anndata==0.11.3",
        "matplotlib==3.10.1",
        "numpy==1.26.4",
        "pandas==2.2.3",
        "scanpy==1.11.0",
        "scipy==1.15.2",
        "torch==2.5.1",
        "torch-geometric==2.6.1",
        "tqdm==4.67.1",
    ],
)