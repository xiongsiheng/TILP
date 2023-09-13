For some required files:
1) pos_examples_idx.json:
It describes the samples used for training. By default (without this file), we use the whole training set. We also do random sampling sometimes. 
2) bg_train.txt:
It describes the background knowledge used for training. By default (without this file), we use the whole training set.
3) bg_test.txt:
It describes the background knowledge used for test. By default (without this file), we use the whole training set.


The complete version can be time-consuming, to accelerate it, you can:
1. random sample some postive examples by setting pos_examples_idx.json (main.py)
2. reduce 'self.num_training_samples', 'self.num_paths_max', 'self.num_path_sampling', 'self.max_rulenum' (Models.py)
3. increase 'num_processes' (all py files)

