# NAC Framework
This repo is the official implementation of "Do Not Train It: A Linear Neural Architecture Search of Graph Neural Networks" (Xu et al., ICML 2023)


## Introduction
The originization of this repo is shown as follow:
```
|-- README.md # short introduction of codes
|-- nac # the python implementation of this project, including NAS Searching Phase and Finetuning Phase
        |-- __init__.py
        |-- controller # NAS updating modules
        |-- lr_scheduler
        |-- model
        |-- optimizer
        |-- solver
        `-- utils
|-- configs  # the typical usage configuration
`-- examples # the configuration for runing experiments
```

## Usage
Go to the workspace dir of examples, run

```
python -W ignore -u -m nac.solver.nac_induc_solver --config="./examples/Tox21/lnac/config_tox21.yaml" --phase train_search
```


## Citation
```
@inproceedings{xu2023do,
  title={Do Not Train It: A Linear Neural Architecture Search of Graph Neural Networks},
  author={Peng Xu and Lin Zhang and Xuanzhou Liu and Jiaqi Sun and Yue Zhao and Haiqin Yang and Bei Yu},
  booktitle={International Conference on Machine Learning},
  year={2023}
}
```