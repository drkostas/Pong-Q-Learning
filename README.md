# Pong Q-Learning

[![GitHub license](https://img.shields.io/badge/license-Apache-blue.svg)](
https://github.com/drkostas/Q-Learning/blob/master/LICENSE)

## Table of Contents

+ [About](#about)
+ [Requirements](#installing)
+ [Running the code](#run)
+ [License](#license)

## About <a name = "about"></a>

The goal of this project is to experiment using Q-learning to beat a Pong game program.
This will include programming the Q-learner, choosing hyper parameters, and quantizing the
state. The assignment comprises of writing a Python-based simulation, and then a report
which analyzes and explains the results.

To get started, follow their respective instructions.

## Requirements <a name = "installing"></a>


Before running the programs, you should first create a conda environment, load it, and install the requirements
like so:

```ShellSession
$ conda create -n q_learning_pong python=3.8
$ conda activate q_learning_pong
$ pip install -r requirements.txt
```


## Running the code <a name = "run"></a>

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
