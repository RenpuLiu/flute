# Federated Representation Learning in the Under-Parameterized Regime (ICML 2024)


Code to reproduce the results for the proposed method FLUTE in our paper [Exploiting Shared Representations for Personalized Federated Learning](https://arxiv.org/pdf/2406.04596.pdf).

Renpu Liu, Cong Shen, Jing Yang

This paper has been accepted at ICML 2024.


## Usage

FLUTE is run using a command of the following form:

`python main_fedrep.py --alg fedrep --dataset [dataset] --num_users [num_users] --model [model] --model [model] --shard_per_user [shard_per_user] --frac [frac] --local_bs [local_bs] --lr [lr] --epochs [epochs] --local_ep [local_ep] --local_rep_ep [local_rep_ep] --gpu [gpu]`



## Citation

Please cite our paper if you use our implementation:

```
@misc{liu2024federated,
      title={Federated Representation Learning in the Under-Parameterized Regime}, 
      author={Renpu Liu and Cong Shen and Jing Yang},
      year={2024},
      eprint={2406.04596},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

# Our code is based on Collins' work

https://github.com/lgcollins/FedRep
