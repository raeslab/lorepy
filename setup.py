import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lorepy",
    version="0.1.0",
    author="Sebastian Proost",
    author_email="sebastian.proost@gmail.com",
    description="Draw Logistic Regression Plots in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sepro/lorepy",
    project_urls={
        "Bug Tracker": "https://github.com/sepro/lorepy/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)