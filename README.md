## TILP: Differentiable Learning of Temporal Logical Rules on Knowledge Graphs
This repository contains the code for the paper [TILP: Differentiable Learning of Temporal Logical Rules on Knowledge Graphs](https://openreview.net/pdf?id=_X12NmQKvX).

## Introduction
We propose TILP, a differentiable framework for temporal logical rules learning. By designing a constrained random walk mechanism and the introduction of temporal operators, we ensure the efficiency of our model. We present temporal features modeling in tKG, e.g., recurrence, temporal order, interval between pair of relations, and duration, and incorporate it into our learning process.

## Commands
To run the code, you need to first set up the environment given in requirements.txt.

It is recommended to use anaconda for installation.

After the installation, you need to create a file folder for experiments. 

The structure of the file folder should be

TILP/
TILP/src/

TILP/data/WIKIDATA12k/

TILP/data/YAGO11k/

TILP/output/found_rules/

TILP/output/found_t_s/

TILP/output/train_weights_tfm/

TILP/output/train_weights/

TILP/output/learned_rules/

TILP/output/explore_res/

TILP/output/rank_dict/


To run the code, simply use the command
```sh
python main.py
```

## Dataset
For some required files:

1) pos_examples_idx.json:

It describes the samples used for training. By default (without this file), we use the whole training set. We also do random sampling sometimes. 

2) bg_train.txt:

It describes the background knowledge used for training. By default (without this file), we use the whole training set.

3) bg_test.txt:

It describes the background knowledge used for test. By default (without this file), we use the whole training set.


The complete version can be time-consuming, to accelerate it, you can:

1) random sample some postive examples by setting pos_examples_idx.json (main.py)

2) reduce 'self.num_training_samples', 'self.num_paths_max', 'self.num_path_sampling', 'self.max_rulenum' (Models.py)

3) increase 'num_processes' (all py files)

## Citation
```
@inproceedings{xiong2022tilp,
  title={TILP: Differentiable Learning of Temporal Logical Rules on Knowledge Graphs},
  author={Xiong, Siheng and Yang, Yuan and Fekri, Faramarz and Kerce, James Clayton},
  booktitle={The Eleventh International Conference on Learning Representations},
  year={2022}
}
```
