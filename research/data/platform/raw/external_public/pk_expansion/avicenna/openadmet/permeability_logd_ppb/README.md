---
license: apache-2.0
language:
- en
tags:
- chemistry
- drug-discovery
- admet
- multitask-learning
- openadmet
---


This is our baseline Caco-2 permeability/LogD/PPB model. It is a **multitask CheMeleon model** trained to predict the following endpoints:

- Caco-2 Permeability Papp A->B
- Caco-2 Permeability Papp B->A
- LogD
- MPPB
- HPPB

> Check out comparative performance on the ExpansionRx dataset here: https://openadmet.ghost.io/zero-shot-expansiorx-admet-predictions/

> **Update Notice**: We have released an updated version of the model (**v2**) featuring enhanced curation of the ChEMBL training set. Key improvements include the exclusion of censored data (non-explicit modifiers like `<` or `>`) to ensure high-fidelity regression and optimized outlier filtering. The legacy version remains accessible via the commit history.

## Pre-requisites
We *highly* recommend you have the Anvil framework from `openadmet-models` installed in an environment (called `openadmet-models`) for ease of use and full utilization of OpenADMET's models. For full documentation, visit our website [here](https://docs.openadmet.org/en/latest/). If you'd like to see some more examples on how to use Anvil, see our demos [here](https://demos.openadmet.org/en/latest/).

### Installation of `openadmet-models`

#### With conda

You can install openadmet-models via our GitHub package. If you want the latest development version, clone the repository and install in editable mode:

```
git clone git@github.com:OpenADMET/openadmet-models.git
```

Set up an environment using the provided files in devtools/conda-envs.
```
cd openadmet-models/
conda env create -f devtools/conda-envs/openadmet-models.yaml
conda activate openadmet-models
pip install -e .
```
If you want to use GPU acceleration, ensure you have the appropriate CUDA toolkit installed and use the openadmet-models-gpu.yaml file instead:
```
conda env create -f devtools/conda-envs/openadmet-models-gpu.yaml
conda activate openadmet-models
pip install -e .
```
#### With Docker
Alternatively, you can also use Docker to spin up a containerized pre-installed environment to run `openadmet-models`. Just be sure you are mounting the correct folder (`./permeability-logd-ppb-chemeleon-baseline`) where you've downloaded the model.

If you're using a gpu, run:
```
docker run -it --user=root --rm  \
	-v ./permeability-logd-ppb-chemeleon-baseline:/home/mambauser/model:rw \
	--runtime=nvidia \
	--gpus \
	all ghcr.io/openadmet/openadmet-models:main
```
Otherwise, for cpu only:
```
docker run -it --user=root --rm  \
	-v ./permeability-logd-ppb-chemeleon-baseline:/home/mambauser/model:rw \
	all ghcr.io/openadmet/openadmet-models:main
```

**IMPORTANT NOTE** You will also need `git lfs` installed.

## Downloading the model

1. After installing Anvil, clone the model repo:
```
git clone https://huggingface.co/openadmet/permeability-logd-ppb-chemeleon-baseline/
```
2. Change to the repo directory. Ensure you have `git lfs` installed for the repo and get the large model files:
```
git lfs install
git lfs pull
```
3. You are now ready to use the model!

## Using the model

**IMPORTANT NOTE:** This model predicts \\(\log_{10}(P_{app})\\) values (on \\(\log_{10}(\text{cm/s})\\)). To get \\(P_{app}\\) values in \\(10^{-6} \text{cm/s}\\), simply backtransform:
$$
P_{app} = 10^{\hat{y}} * 10^{6}
$$

Where \\(\hat{y}\\) is our model prediction. 

For the protein binding endpoints, the model predicts \\(\log_{10}(\% \text{unbound})\\). To get PPB values from %bound, simply subtract \\(100 - \% \text{bound}\\).

We will use this model for inference, to predict endpoint values for a set of molecular compounds unseen to the model. For demonstration purposes, we will be using a small-molecule set from our recent [OpenADMET-ExpansionRx challenge](https://huggingface.co/spaces/openadmet/OpenADMET-ExpansionRx-Challenge), provided in the file `expansion_data_inference.csv`.
You can do this either **inside the docker container** as per the instructions above, or if you have installed openadmet-models on your own computer, you can use the appropriate environment.

The generic command to run our inference pipeline is:
```bash
openadmet predict \
	--input-path <the path to the data to predict on> \
	--input-col <the column of the data to predict on, often SMILES> \
	--model-dir <the anvil_training directory of the model to predict with> \
	--output-csv <the path to an output CSV to save the predictions to> \
	--accelerator <whether to use gpu or cpu, defaults to gpu>
```
You can run this directly in your command line, OR you can use the bash script we've provided, `run_model_inference.sh`.

For our working example, this command becomes:
```bash
openadmet predict \
	--input-path expansion_data_inference.csv \
	--input-col SMILES \
	--model-dir anvil_training/ \
	--output-csv predictions.csv \
	--accelerator cpu
```
You can easily substitute your own set of compounds, simply modify the `--input-path` and `--input-col` arguments for your specific dataset.

In our example, this outputs a file called `predictions.csv` which includes endpoint-specific prediction columns (as `OADMET_PRED_chemprop_{}`) for:

- `caco2_atob_LogPapp`
- `caco2_btoa_LogPapp`
- `logD`
- `mppb_LogUnbound`
- `hppb_LogUnbound`

In this case, `OADMET_STD_chemprop_{}` columns are empty because uncertainty cannot be estimated unless running inference on an ensemble of models. See how to set this option [here](https://demos.openadmet.org/en/latest/demos/04_Ensemble_Model_Training/04_Ensemble_Model_Training_Active_Learning.html).

**IMPORTANT NOTE** If you'd like other examples for how to use our Anvil framework, checkout our demos [here](https://demos.openadmet.org/en/latest/).
