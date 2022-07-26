# TinyML MNIST

## How to prepare build environment

```shell-session
$ brew install direnv
...
$ eval "$(direnv hook bash)"
...
$ brew install anyenv
...
$ yes | anyenv install --init
...
$ eval "$(anyenv init -)"
...
$ anyenv install pyenv
...
$ pyenv install 3.10.5
...
$ python -m pip install --upgrade pip
...
$ python -m venv .venv
...
$ direnv allow
...
$ pip install jupyter notebook
...
$
```

## How to launch Jupyter Notebook

```shell-session
$ jupyter notebook
...
```

## Prepare Arduino IDE

Add the following URL to board manager URL of preference:

- <https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json>

Install libraries:

- EloquentTinyML
