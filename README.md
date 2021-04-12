# CS60075-Team-20-Task-11
Submission for SemEval 2021 Task 11 - NLP Contribution Graph by Team 20

Team members are:

- Siba Smarak Panigrahi (18CS10069)
- Mukul Mehta (18CS10033)
- Aditya Singh (18CS30005)
- Ram Niwas Sharma (18CS10044)
- Rashil Gandhi (18CS30036)

The organization of the code is as follows:
- subtask_A contains code and results for Subtask A of the task
- subtask_B contains code and results for Subtask B of the task

## Obtaining the code and data

First, clone this repository
```
https://github.com/mukul-mehta/CS60075-Team-20-Task-11.git
cd CS60075-Team-20-Task-11
```

In order to obtain the datasets, create a folder called ```datasets``` in the root of the project. Then run the following
to clone the dataset repositories

```
git clone https://github.com/ncg-task/training-data.git datasets/train
git clone https://github.com/ncg-task/trial-data.git datasets/validation
git clone https://github.com/ncg-task/test-data datasets/test
```

If you are downloading the datasets at the different location, change the values in the file ```subtask_A/config.ini```

Create a new virtual environment using your favourite tool and activate it. Once inside the virtualenv, install dependencies using
```
pip install -r requirements.txt
```

## Running the models

To run subtask_A, run the following
```
cd subtask_A
python inference.py --model {baseline, bert-linear, bert-bilstm}
```

This will save the model in the directory specified in the ```config.ini``` file in the same folder

To run subtask_B, run the following
```
cd subtask_B
python main.py
```
