# Lorepy - Development documentation

## Setting up the environment

To recreate the environment used by the devs, you can get a [requirements.txt](./dev/requirements.txt) file that has the
same versions we have been using pinned. To install these after creating a virtual environment use the command below
(from the root of the project)

```bash
pip install -r ./docs/dev/requirements.txt
```


## Running tests

Lorepy is fully covered with unit-tests, to run them you need the pytest package installed (```pip install pytest pytest-cov```).
Next, run the command below to run the test suite. Note: if you use the environment listed above you will get these.

```bash
pytest
```
To enable coverage stats run the command below.

```bash
pytest --exitfirst --verbose --failed-first --cov=src
```

## Deploying on PyPi

### Building the package

To build the source distribution along with a wheel, use the command below. 

```bash
python setup.py sdist bdist_wheel
```

### Push the package to PyPi

**Note** that these commands will upload the code to publicly available platforms, use with caution !

This will require the twine package, install twine using ```pip install twine``` if needed.

You can upload a new build to [TestPyPi] using the command below:

```bash
twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```

Once everything is ready to be uploaded to [PyPi], one more command is necessary:

```bash
twine upload dist/*
```

When prompted for credentials, use `__token__` as the username and the API token generated on [PyPi] as the password.

[TestPyPi]: https://test.pypi.org/
[PyPi]: https://pypi.org/