# Federated Representation Learning in the Under-Parameterized Regime (ICML 2024)

This repository contains the code to reproduce the results for the proposed method FLUTE as presented in our paper:

[Exploiting Shared Representations for Personalized Federated Learning](https://arxiv.org/pdf/2406.04596.pdf) by Renpu Liu, Cong Shen, and Jing Yang.

This paper has been accepted at ICML 2024.

## Usage

To run FLUTE, use the following command template:

```bash
python main_flute.py --alg flute --dataset [dataset] --num_users [num_users] --model [model] --shard_per_user [shard_per_user] --frac [frac] --local_bs [local_bs] --lr [lr] --epochs [epochs] --local_ep [local_ep] --local_rep_ep [local_rep_ep] --server_update [server_update]
```

### Example Command

```bash
python main_flute.py --alg flute --dataset cifar10 --num_classes 10 --num_users 100 --model cnn --shard_per_user 5 --frac 0.1 --local_bs 10 --lr 0.01 --epochs 100 --local_ep 10 --local_rep_ep 1 --server_update 1
```

## Citation

If you use our implementation, please cite our paper:

```bibtex
@misc{liu2024federated,
      title={Federated Representation Learning in the Under-Parameterized Regime}, 
      author={Renpu Liu and Cong Shen and Jing Yang},
      year={2024},
      eprint={2406.04596},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Acknowledgements

Our code is based on Collins' work, available at [FedRep GitHub Repository](https://github.com/lgcollins/FedRep).
