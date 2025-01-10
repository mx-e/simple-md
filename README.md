
### This is a fork of [this](https://github.com/marvinsxtr/ml-project-template) repo. Functionality is similar - a few weird behaviours in edge cases were fixed but most changes are for UX and flavor (see below for details). Check out the repo above to see a similar (possibly better) take on the same problem.


This is a template project for ML experimentation using wandb, hydra-zen, submitit on a Slurm cluster using Docker and Apptainer for containerization.

**NOTE**: This template is optimized for the specific setup of the ML Group cluster but may be easily adapted to similar settings.

## Highlights

* Apptainer containerization using [pip-poetry](https://python-poetry.org) to configure python evironment
* Logging and visualizations via [Weights and Biases](https://wandb.com)
* Reproducibility and modular type-checked configs via [hydra-zen](https://github.com/mit-ll-responsible-ai/hydra-zen)
* Submit Slurm jobs and sweeps directly from Python via [submitit](https://github.com/facebookincubator/submitit)


## Main differences to [main repo](https://github.com/marvinsxtr/ml-project-template)

* No interactive shell in the container needed - everything is run by using the script itself as an executable
* Freely change the main function - the addon features' configuration is only added on top of your own config
* Using poetry as package manager instead of conda - dependencies are stored in the pyproject.toml
* Using apptainer to be able to update the container directly on the cluster
* Simplified structure - managed to get rid of some code by using built-in Hydra functionality


## Setup

### Step 1: Env Vars
To be able to run Slurm from within Apptainer, you first have to add the following lines to your `.zshrc`/`.bashrc` file:

```bash
export APPTAINER_BIND=/opt/slurm-23.2,/opt/slurm,/etc/slurm,/etc/munge,/var/log/munge,/var/run/munge,/lib/x86_64-linux-gnu
export APPTAINERENV_APPEND_PATH=/opt/slurm/bin:/opt/slurm/sbin
```

### Step 2 (optional): Setup Aliases
The following aliases for interacting with Apptainer are recommended:

```bash
alias rebuild='apptainer build --nv container.sif container.def'
alias add-dep='apptainer run --nv container.sif poetry --no-cache add --lock'
alias remove-dep='apptainer run --nv container.sif poetry --no-cache remove --lock'
```

### Step 3: Change container path
In the first line of each script in the ```scripts``` directory, change file path the first line to the container's absolute path:

```python
./scripts/train.py                  edit this path \/
#! /usr/bin/env -S apptainer exec /home/maxi/TEMP/ml-project-template/container.sif python 

from loguru import logger

from conf.base_conf import configure_main, BaseConfig
from lib.utils.run import run

...
```
### Step 4: Configure WandB logging
Logging to WandB is optional for running local jobs but mandatory for jobs submitted to the cluster.

WandB is enabled by specifying an API key, the project and entity. Rename `example.env` to `.env` file in the root of the repository. Fill in your information.

### Step 5: Build the container
Build the container using the `rebuild` alias above - if there are any errors try deleting the poetry.lock file and repeat.

## Run
Run the script with 
```bash
./scripts/train.py
```
This will work with any path as long as the script has the first line pointing to the apptainer above. Just note that Hydra will generate the output dir relative to the current working directory, so better be consistent.
Hydra should automatically generate a `config.yaml` in `./outputs/<date>/<time>/.hydra`. 

To log to WandB, add `cfg/wandb=log`:

```bash
./scripts/train.py cfg/wandb=log
```

In order to use WandB in offline mode, add `cfg.wandb.mode=offline`:

```bash
./scripts/train.py cfg/wandb=log cfg.wandb.mode=offline
```

### Single Job

To run the command as a job in the cluster, run

```bash
./scripts/train.py cfg/job=run
```

This will automatically add WandB logging for you. See `src/configs/runs/base.py` to configure the job to your needs.

### Distributed Sweep

Run a sweep over two seeds using multiple nodes:

```bash
./scripts/train.py cfg/job=sweep
```

This will automatically add WandB logging for you. See `src/configs/runs/base.py` to configure the sweep to your needs.

## Edit main function

You can make changes to the main function's signature or rename the function. Hydra-zen will automatically detect all arguments with standard types values as long as you give a default value.
You can also add config groups and all other hydra-zen functionality.

#### **However, you should not remove or rename the first argument as it is needed to configure sweeps, wandb, etc.** 

(You can if you know the code well, but be warned! Be careful making changes to this - in a very unlucky case changing BaseConfigs structure can lead to recursive runs on your infratructure!)


```python
#! /usr/bin/env -S apptainer exec /home/maxi/TEMP/ml-project-template/container.sif python

from loguru import logger

from conf.base_conf import configure_main, BaseConfig
from lib.utils.run import run


@configure_main
def train(
    cfg: BaseConfig,  # you must keep this argument <----
    bar: int = 42,
    foo: str = "hello",
    jup: bool = False,
    test: float = 2.2,
) -> None:
    logger.info("Running main function.")
    logger.info(f"Config: bar={bar}, foo={foo}, jup={jup}")
    logger.info(f"BaseConfig: {cfg}")


if __name__ == "__main__":
    run(train)

```
Of course you can also add more scripts - to work they only need:
1. The first line (`#! /usr/bin/env -S apptainer exec /home/maxi/TEMP/ml-project-template/container.sif python`)
2. The `configure_main` decorator over the main function
3. Running the main function with the `run` function


## Edit dependencies
Edit dependencies in the root dir of the repo with the aliases defined above, e.g.:
```bash
add-dep wandb
remove-dep numpy
```
After editing dependencies rebuild using the ```rebuild``` alias.


### Legacy version

This template is based on a [previous example project](https://github.com/mx-e/example_project_ml_cluster).

## Dataset Setup

### QCML

Coming

### MD17/MD22/sGDML

In order to download the sGDML datasets (which contain e.g. MD17/MD22), activate the poetry environment and execute this in the directory where you want to download the data:
```bash
poetry run sgdml-get dataset all
```
