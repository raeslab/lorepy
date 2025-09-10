from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="lorepy",
    version="0.4.4",
    author="Sebastian Proost",
    author_email="sebastian.proost@gmail.com",
    description="Draw Logistic Regression Plots in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/raeslab/lorepy/",
    project_urls={
        "Bug Tracker": "https://github.com/raeslab/lorepy/issues",
    },
    install_requires=[
        "matplotlib>=3.4.1",
        "numpy>=1.20.2",
        "pandas>=1.2.4",
        "scikit-learn>=1.5.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    license="Creative Commons Attribution-NonCommercial-ShareAlike 4.0. https://creativecommons.org/licenses/by-nc-sa/4.0/",
    packages=find_packages("src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
)
