import setuptools
import os

loc = os.path.abspath(os.path.dirname(__file__))

setuptools.setup(
    name="ensemble_testbed",
    version="0.0.1",
    author="Kelly Buchanan",
    author_email="ekb2154@columbia.edu",
    description="ensemble_tested",
    long_description="",
    long_description_content_type="",
    url="https://github.com/ekellbuch/ensemble_testbed",
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={},
    classifiers=["License :: OSI Approved :: MIT License"],
    python_requires=">=3.6",
)
