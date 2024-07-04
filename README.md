## [ICLR 23] TILP: Differentiable Learning of Temporal Logical Rules on Knowledge Graphs
This repository contains the code for the paper [TILP: Differentiable Learning of Temporal Logical Rules on Knowledge Graphs](https://openreview.net/pdf?id=_X12NmQKvX).

<p align="center">
  <img src='https://github.com/xiongsiheng/TILP/blob/main/misc/Task.png' width=600>
</p>

A follow-up work for **time prediction** over temporal knowledge graphs via **logical reasoning** is available [here](https://github.com/xiongsiheng/TEILP).

## Introduction
We propose TILP, a differentiable framework for temporal logical rules learning. By designing a constrained random walk mechanism and the introduction of temporal operators, we ensure the efficiency of our model. We present temporal features modeling in tKG, e.g., recurrence, temporal order, interval between pair of relations, and duration, and incorporate it into our learning process.

**Rule 1:**

$$
\text{memberOf}\left(E_1,E_4,I_4\right) \leftarrow \text{memberOf}\left(E_1,E_2,I_1\right) \wedge \text{memberOf}^{-1}\left(E_2,E_3,I_2\right) 
$$

$$
\wedge \text{memberOf}\left(E_3,E_4, I_3\right)\wedge^3_{i=1}(\wedge^4_{j=i+1} \text{touching}(I_i,I_j))
$$

**Grounding:** $E_1$ = Somalia, $E_2$ = International Development Association, $E_3$ = Kingdom of the Netherlands, $E_4$ = International Finance Corporation, $I_1$ = $[1962, \text{present}]$, $I_2$ = $[1961, \text{present}]$, $I_3$ = $[1956, \text{present}]$, $I_4$ = $[1962, \text{present}]$.

**Rule 2:**

$$
\text{receiveAward}\left(E_1,E_4,I_4\right) \leftarrow \text{nominatedFor}\left(E_1, E_2, I_1\right) \wedge \text{nominatedFor}^{-1}\left(E_2,E_3,I_2\right) 
$$

$$
\wedge \text{receiveAward}\left(E_3,E_4,I_3\right)\wedge \text{before}(I_1,I_2)\wedge^2_{i=1} \text{after}(I_i,I_3) \wedge^3_{j=1} \text{before}(I_j,I_4)
$$

**Grounding:** $E_1$ = ZDF, $E_2$ = International Emmy Award for best drama series, $E_3$ = DR, $E_4$ = Peabody Awards, $I_1$ = $[2005, 2005]$, $I_2$ = $[2009, 2009]$, $I_3$ = $[1997, 1997]$, $I_4$ = $[2013, 2013]$.



## Quick Start
To run the code, you need to first set up the environment given in requirements.txt.

It is recommended to use anaconda for installation.

After the installation, you need to create a file folder for experiments. 

The structure of the file folder should be like
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
    ├── found_paths/
    ├── found_time_gaps/
    ├── train_weights/
    ├── train_weights_tfm/
    ├── learned_rules/
    └── rank_dict/
```

To run the code, simply use the command. 

All the settings and corresponding explanations are provided as in-context comments.
```sh
cd src
python main.py
```

Print rules:
```sh
# Find the rules and scores in output/learned_rules/{$dataset name}_all_rules_{$query relation}.json.
# Format: {rule length: [{support num: int, rule: list, score: float}]}
#          rule: a list of relations and temporal relations (Example in YAGO: query relation 0, rule length: 3, rule: [1, 16, 0, 1, 1, -1, 0, 1, 1]; Translation: wasBornIn(x, y, I_q) <- worksAt(x, e1, I_1) and graduatedFrom^{-1}(e1, e2, I_2) and wasBornIn(e2, y, I_3) and after(I_q, I_1) and after(I_q, I_2) and before(I_q, I_3) and touching(I_1, I_2) and after(I_1, I_3) and after(I_2, I_3))
```

## Contact
If you have any inquiries, please feel free to raise an issue or reach out to sxiong45@gatech.edu.

## Citation
```
@inproceedings{xiongtilp,
  title={TILP: Differentiable Learning of Temporal Logical Rules on Knowledge Graphs},
  author={Xiong, Siheng and Yang, Yuan and Fekri, Faramarz and Kerce, James Clayton},
  booktitle={The Eleventh International Conference on Learning Representations}
}
```
