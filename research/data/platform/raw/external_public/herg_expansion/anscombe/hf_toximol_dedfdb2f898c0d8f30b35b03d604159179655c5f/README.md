---
license: mit
task_categories:
- tabular-classification
- tabular-regression
task_ids:
- multi-class-classification
- tabular-single-column-regression
multilinguality:
- monolingual
size_categories:
- 1K<n<10K
source_datasets:
- original
language:
- en
tags:
- chemistry
- toxicity
- molecular-design
- SMILES
- drug-discovery
- benchmark
- multimodal
- structure-activity-relationship
pretty_name: "ToxiMol: A Benchmark for Structure-Level Molecular Detoxification"
dataset_info:
  features:
    - name: task
      dtype: string
    - name: id
      dtype: int64
    - name: smiles
      dtype: string
    - name: image_path
      dtype: string
  splits:
    - name: test
      num_examples: 660
  configs:
    - config_name: ames
      data_files: "ames/*"
    - config_name: carcinogens_lagunin
      data_files: "carcinogens_lagunin/*"
    - config_name: clintox
      data_files: "clintox/*"
    - config_name: dili
      data_files: "dili/*"
    - config_name: herg
      data_files: "herg/*"
    - config_name: herg_central
      data_files: "herg_central/*"
    - config_name: herg_karim
      data_files: "herg_karim/*"
    - config_name: ld50_zhu
      data_files: "ld50_zhu/*"
    - config_name: skin_reaction
      data_files: "skin_reaction/*"
    - config_name: tox21
      data_files: "tox21/*"
    - config_name: toxcast
      data_files: "toxcast/*"
---

# ToxiMol: A Benchmark for Structure-Level Molecular Detoxification

[![arXiv](https://img.shields.io/badge/arXiv-paper-red.svg)](https://arxiv.org/pdf/2506.10912)
[![Dataset](https://img.shields.io/badge/🤗%20Hugging%20Face-ToxiMol-blue)](https://huggingface.co/datasets/DeepYoke/ToxiMol-benchmark)

## 🔥 News

- 🏆 **ToxiMol** has been accepted as an **Oral** presentation at the **32nd SIGKDD Conference on Knowledge Discovery and Data Mining (KDD 2026), AI for Sciences Track**.

## Overview

**ToxiMol** is the first comprehensive benchmark for **molecular toxicity repair** tailored to general-purpose **Multimodal Large Language Models (MLLMs)**. This is the dataset repository for the paper "Breaking Bad Molecules: Are MLLMs Ready for Structure-Level Molecular Detoxification?".

## Key Features

### 🧬 Comprehensive Dataset
- **660 representative toxic molecules** spanning diverse toxicity mechanisms and varying granularities
- **11 primary toxicity repair tasks** based on [Therapeutics Data Commons (TDC) platform](https://tdcommons.ai/single_pred_tasks/tox/)
- **Multi-granular coverage**: Tox21 (12 sub-tasks), ToxCast (10 sub-tasks), and 9 additional datasets
- **Multimodal inputs**: SMILES strings + 2D molecular structure images rendered using RDKit

### 🎯 Challenging Task Definition
The molecular toxicity repair task requires models to:
1. **Identify potential toxicity endpoints** from molecular structures
2. **Interpret semantic constraints** from natural language descriptions
3. **Generate structurally similar substitute molecules** that eliminate toxic fragments
4. **Satisfy drug-likeness and synthetic feasibility** requirements

### 📊 Systematic Evaluation
- **ToxiEval framework**: Automated evaluation integrating toxicity prediction, synthetic accessibility, drug-likeness, and structural similarity
- **Comprehensive analysis**: Evaluation of ~30 mainstream MLLMs with ablation studies
- **Multi-dimensional metrics**: Success rate analysis across different evaluation criteria and failure modes

## Dataset Structure

### Task Overview
| Dataset | Task Type | # Molecules | Description |
|---------|-----------|-------------|-------------|
| **AMES** | Binary Classification | 60 | Mutagenicity testing |
| **Carcinogens** | Binary Classification | 60 | Carcinogenicity prediction |
| **ClinTox** | Binary Classification | 60 | Clinical toxicity data |
| **DILI** | Binary Classification | 60 | Drug-induced liver injury |
| **hERG** | Binary Classification | 60 | hERG channel inhibition |
| **hERG_Central** | Binary Classification | 60 | Large-scale hERG database with integrated cardiac safety profiles |
| **hERG_Karim** | Binary Classification | 60 | hERG data from Karim et al. |
| **LD50_Zhu** | Regression (log(LD50) < 2) | 60 | Acute toxicity |
| **Skin_Reaction** | Binary Classification | 60 | Adverse skin reactions |
| **Tox21** | Binary Classification (12 sub-tasks) | 60 | Nuclear receptors, stress response pathways, and cellular toxicity mechanisms (ARE, p53, ER, AR, etc.) |
| **ToxCast** | Binary Classification (10 sub-tasks) | 60 | Diverse toxicity pathways including mitochondrial dysfunction, immunosuppression, and neurotoxicity |

### Data Structure
Each entry contains:
```json
{
  "task": "string",      // Toxicity task identifier
  "id": "int",           // Molecule ID
  "smiles": "string",    // SMILES representation
  "image": "binary" // 2D molecular structure image binary
}
```

<!-- ## Usage -->
<!-- 
### Loading the Dataset

```python
from datasets import load_dataset

# Load a specific subdataset
ames_dataset = load_dataset("DeepYoke/ToxiMol-benchmark", "ames")
tox21_dataset = load_dataset("DeepYoke/ToxiMol-benchmark", "tox21")

# Access the data
for example in ames_dataset['test']:
    print(f"Task: {example['task']}")
    print(f"ID: {example['id']}")
    print(f"SMILES: {example['smiles']}")
    print(f"Image: {example['image_path']}")
    print("-" * 50)
``` -->

## Available Subdatasets
```python
subdatasets = [
    "ames", "carcinogens_lagunin", "clintox", "dili", 
    "herg", "herg_central", "herg_karim", "ld50_zhu", 
    "skin_reaction", "tox21", "toxcast"
]

# Load all datasets
datasets = {}
for name in subdatasets:
    datasets[name] = load_dataset("DeepYoke/ToxiMol-benchmark", data_dir=name)
```

## Experimental Results

Our systematic evaluation of ~30 mainstream MLLMs reveals:
- **Current limitations**: Overall success rates remain relatively low across models
- **Emerging capabilities**: Models demonstrate initial potential in toxicity understanding, semantic constraint adherence, and structure-aware molecule editing
- **Key challenges**: Structural validity, multi-dimensional constraint satisfaction, and failure mode attribution

## Citation

If you use this dataset in your research, please cite:

```bibtex
@misc{lin2026breakingbadmoleculesmllms,
      title={Breaking Bad Molecules: Are MLLMs Ready for Structure-Level Molecular Detoxification?}, 
      author={Fei Lin and Ziyang Gong and Cong Wang and Tengchao Zhang and Yonglin Tian and Yining Jiang and Ji Dai and Chao Guo and Xiaotong Yu and Xue Yang and Gen Luo and Fei-Yue Wang},
      year={2026},
      eprint={2506.10912},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.10912}, 
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.