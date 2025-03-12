# How simple can you make MD? 

This is the code for [this paper](https://arxiv.org/abs/2503.01431). 

## How to run

This code is written to run on a slurm cluster. It also runs elsewhere but there will be differences.

Before running build the singularity container:
```bash
apptainer build --nv container.sif container.def
```

You can then run a script using, for instance:
```
apptainer run --nv container.sif python ./scripts/train.py
```
Make sure you bind the needed directories (data etc.)

To override parameters use hydra CLI syntax, e.g.:
```
apptainer run --nv container.sif python ./scripts/train.py train.lr=0.0001 train.total_steps=100000 
```


### QCML Data

See [this](https://www.nature.com/articles/s41597-025-04720-7) paper on how to get the QCML dataset in array_record format.







