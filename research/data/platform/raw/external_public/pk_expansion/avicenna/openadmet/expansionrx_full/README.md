---
license: cc-by-4.0
license_link: LICENSE.md
tags:
- admet
- drug discovery
pretty_name: openadmet-expansion-challenge-data
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files:
  - split: train
    path: expansion_data_train.csv
  - split: test
    path: expansion_data_test.csv
- config_name: raw
  data_files:
  - split: train
    path: expansion_data_raw.csv
extra_gated_fields:
  First name: text
  Last name: text
  Company: text
  Work email: text
  I want to use this dataset for: text
---
# OpenADMET-ExpansionRx Challenge FULL dataset

This is the full dataset used in the OpenADMET-ExpansionRx blind challenge, which finalized in January 19th, 2026. 

Originally split in a train and blinded test set, we now release the **full** dataset, which contains real-work ADMET data from a recently prosecuted series of drug discovery campaigns by Expansion Therapeutics on RNA mediated diseases. 
While optimising candidate molecules for their preclinical programs Expansion collected a variety of ADMET data for off-targets and properties of interest in the traditional game of “whack-a-mole” familiar to all drug hunters. 

We are extremely greatful with Expansion Theraputics for releasing this high-quality dataset, and we hope it will help the drug discovery community develop new predictive models and improve existing ADMET-prediction methods. In the words of Jon Ainsley, from the ExpansionRx team:


> When we launched this challenge, we asked the scientific community to put our data to work - and honestly, they delivered beyond anything I imagined. Over 370 participants brought creativity, rigor, and genuine collaboration to a problem that matters deeply, not only to Expansion, but the wider drug discovery community. This is what's possible when real project datasets meet open science. It's been remarkable to see the community's ingenuity on full display, approaches I hadn't considered, new methods shared openly, and the state of the art brought into the open where everyone can learn from it.
Now, with the full dataset released, we pass the baton to the broader community. Build on it, benchmark against it, prove us wrong about what's predictable. Every improvement gets us a step closer to a future where ADME becomes straightforward.
To others with data to share: consider publishing what you can and give the community better problems to solve. The more real-world data we collectively put out there, the faster we make drug discovery simpler, and the sooner patients benefit."

Here, you will find two versions of the dataset: One is the dataset in "ML-ready" format where only in-range measurements are included. The "raw" dataset is also available that includes measurements with out of range modifiers e.g ">" or ">".

## Endpoints 


Participants were tasked with solving real-world ADMET prediction problems ExpansionRx faced during lead optimization. Specifically, predicting the ADMET properties of late-stage molecules based on earlier-stage data from the same campaigns. The dataset encompasses nine (9) crucial endpoints:

* LogD
* Kinetic Solubility KSOL: uM
* Mouse Liver Microsomal (MLM) CLint: mL/min/kg
* Human Liver Microsomal (HLM) Clint: mL/min/kg
* Caco-2 Efflux Ratio
* Caco-2 Papp A>B (10^-6 cm/s)
* Mouse Plasma Protein Binding (MPPB): % Unbound
* Mouse Brain Protein Binding (MBPB): % Unbound
* Mouse Gastrocnemius Muscle Binding (MGMB): % Unbound
* An additional ednpoint **Rat Liver Microsomal (RLM) stability CLint (mL/min/kg)**, which was not part of the challenge, is included in the *raw* dataset.

## Example usage

Using Hugging Face datasets, with control over splits:

```python
from datasets import load_dataset
# train
train_df = load_dataset("openadmet/openadmet-expansionrx-challenge-data", split="train").to_pandas()

# test
test_df = load_dataset("openadmet/openadmet-expansionrx-challenge-data", split="test").to_pandas()

# both splits combined
combined_df = load_dataset("openadmet/openadmet-expansionrx-challenge-data", split="train+test").to_pandas()
```

Using Pandas:

```python
import pandas as pd
# train set
df = pd.read_csv("hf://datasets/openadmet/openadmet-expansionrx-challenge-data/expansion_data_train.csv")
```

Or for the raw dataset 

```python
import pandas as pd
df = pd.read_csv("hf://datasets/openadmet/openadmet-expansionrx-challenge-data/expansion_data_raw.csv")
```