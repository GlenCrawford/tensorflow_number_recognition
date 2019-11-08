# Number recognition with a neural network with Tensorflow

Build and train a Neural Network with Tensorflow to recognize handwritten digits with ~92% accuracy.

Adapted from [this tutorial by Ellie Birbeck](https://www.digitalocean.com/community/tutorials/how-to-build-a-neural-network-to-recognize-handwritten-digits-with-tensorflow), updated for later Tensorflow versions and with documentation inlined as comments.

## Setup

### Setting up Python on a Mac

The MacOS built-in Python is 2.7.X, which is old and soon to be deprecated. Like Ruby, you can use a version manager tool to manage and use multiple versions instead of the default one.

* Install [pyenv](https://github.com/pyenv/pyenv), a Python version manager (like RVM): `brew install pyenv`
* Install the latest stable of Python into pyenv: `pyenv install 3.7.4`
* Set the global default Python version: `pyenv global 3.7.4`
* To load pyenv into your bash shell sessions: `echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.bash_profile`
* And load it into your current session: `source ~/.bash_profile`
* Check that pyenv hijacks Python in the shell: `which python`
* Check the active version is what you installed: `python -V`
* And check that you have [pip](https://pypi.org/project/pip/) (the Python package manager): `pip -V`

### Set up the project

cd to where you keep your projects, then:

```bash
mkdir tensorflow_number_recognition
cd tensorflow_number_recognition
git init
```

### Initialize a virtual environment for this project

Using [venv](https://docs.python.org/3/library/venv.html), the built-in module for virtual environments in Python.

```bash
python -m venv tensorflow_number_recognition
touch .gitignore
echo 'tensorflow_number_recognition' > .gitignore
source tensorflow_number_recognition/bin/activate
```

### Install dependencies using pip

```bash
touch dependencies.txt
```

Add to the file:

```
image==1.5.20
numpy==1.14.5
tensorflow==1.14.0
```

And install the dependencies:

```bash
pip install -r dependencies.txt
```


## Write the Python script

```bash
touch main.py
```

See main.py file in this repo.

## Run the Python script to train and test the network

```bash
python main.py
```

## Test the network manually with real input

```bash
curl -O https://raw.githubusercontent.com/do-community/tensorflow-digit-recognition/master/test_img.png
```

Then uncomment the last section of main.py and run it again.
