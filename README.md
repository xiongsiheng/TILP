## TILP: Differentiable Learning of Temporal Logical Rules on Knowledge Graphs
This repository contains the code for the paper [TILP: Differentiable Learning of Temporal Logical Rules on Knowledge Graphs](https://openreview.net/pdf?id=_X12NmQKvX).

## Introduction
We propose TILP, a differentiable framework for temporal logical rules learning. By designing a constrained random walk mechanism and the introduction of temporal operators, we ensure the efficiency of our model. We present temporal features modeling in tKG, e.g., recurrence, temporal order, interval between pair of relations, and duration, and incorporate it into our learning process.

## How to run
To run the code, you need to first set up the environment given in requirements.txt.

It is recommended to use anaconda for installation.

After the installation, you need to create a file folder for experiments. 

The structure of the file folder should be
```sh
TILP/
│
├── src/
│
├── data/
│   ├── WIKIDATA12k/
│   └── YAGO11k/
│
└── output/
    ├── found_rules/
    ├── found_t_s/
    ├── train_weights_tfm/
    ├── train_weights/
    ├── learned_rules/
    ├── explore_res/
    └── rank_dict/
```

To run the code, simply use the command
```sh
cd src
python main.py
```


## Citation
```
@inproceedings{xiong2022tilp,
  title={TILP: Differentiable Learning of Temporal Logical Rules on Knowledge Graphs},
  author={Xiong, Siheng and Yang, Yuan and Fekri, Faramarz and Kerce, James Clayton},
  booktitle={The Eleventh International Conference on Learning Representations},
  year={2022}
}
```
