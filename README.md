# Pong Q-Learning

[![GitHub license](https://img.shields.io/badge/license-Apache-blue.svg)](
https://github.com/drkostas/Q-Learning/blob/master/LICENSE)

## Table of Contents

+ [About](#about)
+ [Getting Started](#getting_started)
    + [Prerequisites](#prerequisites)
+ [Installing, Testing, Building](#installing)
+ [Running locally](#run_locally)
+ [License](#license)

## About <a name = "about"></a>

The goal of this project is to experiment using Q-learning to beat a Pong game program.
This will include programming the Q-learner, choosing hyper parameters, and quantizing the
state. The assignment comprises of writing a Python-based simulation, and then a report
which analyzes and explains the results.

To get started, follow their respective instructions.

## Getting Started <a name = "getting_started"></a>

These instructions will get you a copy of the project up and running on your local machine for
development and testing purposes. See deployment for notes on how to deploy the project on a live
system.

### Prerequisites <a name = "prerequisites"></a>

You need to have a machine with Python > 3.8 
and any Bash based shell (e.g. zsh) installed.

```ShellSession

$ python3.8 -V
Python 3.8

$ echo $SHELL
/usr/bin/zsh

```


## Installing, Testing, Building <a name = "installing"></a>


Before running the programs, you should first create a conda environment, load it, and install the requirements
like so:

```ShellSession
$ conda create -n q_learning_pong python=3.8
$ conda activate q_learning_pong
$ pip install -r requirements.txt
```


## Running the code <a name = "run_locally"></a>

In order to run the code, first, make sure you are in the correct virtual environment:

```ShellSession
$ conda activate q_learning_pong

$ which python
/home/drkostas/anaconda3/envs/q_learning_pong/bin/python

```

Now, in order to train the agent you can call the [Q_Learning.py](Q_Learning.py)
directly.

```ShellSession
$ python Q_Learning.py q a e n c f
```

Then, to test the agent you can call the [runEpisode.py](runEpisode.py)
directly.

```ShellSession
$ python runEpisode.py q f
```



## License <a name = "license"></a>

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.
