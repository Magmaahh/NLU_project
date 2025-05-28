# NLU Course Project - Part 2 NLU (Lab 5)
This folder contains the code I wrote to complete Part 2.B of
the NLU course project at the University of Trento, focused on the joint task of 
**intent classification** and **slot filling**.

For more details about tasks to solve, implementations and results, please refer to the provided report ([NLU_report.pdf](../NLU_report.pdf)). 

## Code usage
The code loads and fine-tunes a pre-trained BERT model (*bert-base-uncased*) on intent classification and slot filling using the ATIS dataset.

It provides both a training and a testing mode:

- Training mode is used to train one of proposed model configurations from scratch.\
Models trained this way are then saved in an apposite folder, "training_results/models".

- Testing mode is used to evaluate the saved best models for each configuration on the test set, which are contained in an apposite folder, "[bin](./bin)".

**Be sure to run all commands from the project root directory** (i.e., the folder containing this README) to ensure all relative paths resolve correctly.

To correctly use the code for both modes, please refer to the sections below.

### Training mode
To use the code in training mode, run:

```bash
python3 main.py
```

Then, type "*train*"; type "*y*" if you want to change any hyperparameter value, and follow the instructions to do so; type "*n*" otherwise.

### Testing mode
To use the code in testing mode, run:

```bash
python3 main.py
```

Then, type "*test*".