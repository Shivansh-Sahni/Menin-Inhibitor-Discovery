# Source and reproducibility ledger

Audit date: 2026-08-07. Counts and metrics in the matrices are tied to a primary paper, official supporting information, or an official repository snapshot. Secondary/search snippets were used only to locate primary artifacts. `NR` is used where an exact primary value was not recovered.

## Primary literature and artifacts

| Model/method | Primary source | Official artifact checked | Evidence used |
|---|---|---|---|
| MaxQSARing | [JPHA paper](https://doi.org/10.1016/j.jpha.2025.101411), [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12756696/) | [official repository](https://github.com/iipharma/maxqsaring), commit `c814249924ef553dffabe2ac5c8520c07ab59f3a` | development/external counts, split, MCC/accuracy, representation search, MIT license |
| Transformer_Morgan | [JPHA paper](https://doi.org/10.1016/j.jpha.2025.101263), [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12446640/) | no frozen official repository verified | paper counts, 80/20 split, architecture, random/external metrics; arithmetic inconsistency retained explicitly |
| XGB+ISE | [Scientific Reports paper and SI](https://doi.org/10.1038/s41598-025-99766-3) | official SI2 CSV inspected locally during audit | exact train/internal/external rows and labels; full-set versus 64%-coverage metrics; descriptor/ISE selection procedure |
| HERGAI | [Journal of Cheminformatics paper](https://doi.org/10.1186/s13321-025-01063-8) | [official repository](https://github.com/vktrannguyen/HERGAI) | exact class/split counts, docking/PLEC method, test metrics, repository/archive/license status |
| hERGBoost | [Computers in Biology and Medicine paper](https://doi.org/10.1016/j.compbiomed.2024.109416), [PubMed](https://pubmed.ncbi.nlm.nih.gov/39550914/), [official Korean patent publication](https://patents.google.com/patent/KR20260004987A/ko) | [author web-server URL](http://ssbio.cau.ac.kr/software/hergboost); frozen download not recovered | 10,798 study count, 706 external count, external regression/classification metrics, and extreme-potency error analysis; unverified internal CV numbers were excluded from the final matrices |
| hERGAT | [Journal of Cheminformatics paper](https://doi.org/10.1186/s13321-025-00957-x) | [official repository](https://github.com/bmil-jnu/hERGAT), audited at `7a706ff084faffbd9782fa4075d3e099e62f6533` | exact class counts, split, paper metrics, current-repository metric drift |
| hERG-MFFGNN | [BioMedical Engineering OnLine paper](https://doi.org/10.1007/s12539-025-00768-6), [PubMed](https://pubmed.ncbi.nlm.nih.gov/40983850/) | [official repository](https://github.com/zhaoqi106/hERG-MFFGNN), commit `bb8cca8d83811702ceb56675a928353948fb7674` | repository-exact benchmark/external counts; code-level fixed test fold and test-aware checkpoint selection; paper AUC/accuracy |
| TDMFLSGAT | [ACS Omega paper](https://doi.org/10.1021/acsomega.5c10745), [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC13019261/) | [official repository](https://github.com/BeidjaCheikh/TDMFLSGAT_data_and_code), audited at `b5019e06bca0a33b53cbe70e88a7cd350afd8df1` | fivefold metrics; workbook count; empty test sheet; fixed-split index; missing/nonportable artifact observations |
| MultiCTox | [JCIM paper](https://doi.org/10.1021/acs.jcim.5c00022) | [official repository](https://github.com/3505675604/MultiCTox), commit `9c079268f5fd07a4132997c5c5f828b9e8a30005` | repository-exact development/split/external counts and pIC50 threshold; no exact paper headline transcribed without full tables |
| CToxPred2 | [JCIM paper](https://doi.org/10.1021/acs.jcim.4c01102) | [official CToxPred repository](https://github.com/issararab/CToxPred) | semisupervised/multi-channel design and approximate unlabeled scale; labeled counts/metrics left `NR` in this audit |
| Mixture-of-Experts cardiotoxicity | [Journal of Cheminformatics paper](https://doi.org/10.1186/s13321-025-01072-7) | [official repository](https://github.com/EdoardoVigano/MoECardiotoxicity) | global holdout and hERG sensitivity/specificity; exact hERG training count left `NR` |
| CardioSafe v1.1 | [2026 preprint](https://doi.org/10.64898/2026.05.06.723181) | [official benchmark repository](https://github.com/AppliedScientific/CardioSafe-benchmark), commit `a481ffb2ea2e53acf944d2e2ad961162d02727db` | v1.1 labels/splits/metrics, v1.0 leakage audit and correction, licenses, incomplete training-loader caveat; explicitly not peer-reviewed |
| Karim large HTS | [Current Research in Toxicology paper](https://doi.org/10.1016/j.crtox.2023.100121), [PubMed](https://pubmed.ncbi.nlm.nih.gov/37701072/) | no official runnable snapshot audited | exact 203,853/87,366 counts and reported model AUCs |
| ActFound | [Nature Machine Intelligence paper](https://doi.org/10.1038/s42256-024-00876-w) | [official repository](https://github.com/HFUT-ML/ActFound) | 1.6-million/35,644-assay scale and pairwise meta-learning role; no hERG metric claimed |
| UQ4DD censored adapter | [Artificial Intelligence in the Life Sciences paper](https://doi.org/10.1016/j.ailsci.2025.100128) | [official repository](https://github.com/MolecularAI/uq4dd) | censored NLL/MSE method, temporal 15-assay proprietary evaluation; no public hERG performance claimed |

## Repository audit commands

The following read-only commands were run against temporary clones; the temporary paths are not project dependencies.

```bash
git clone --depth 1 https://github.com/3505675604/MultiCTox.git /private/tmp/MultiCTox-v2
git clone --depth 1 https://github.com/zhaoqi106/hERG-MFFGNN.git /private/tmp/hERG-MFFGNN-v2
git clone --depth 1 https://github.com/iipharma/maxqsaring.git /private/tmp/maxqsaring-v2
git -C /private/tmp/MultiCTox-v2 rev-parse HEAD
git -C /private/tmp/hERG-MFFGNN-v2 rev-parse HEAD
git -C /private/tmp/maxqsaring-v2 rev-parse HEAD
wc -l /private/tmp/hERG-MFFGNN-v2/dataset/*.csv
awk -F, 'NR>1 {gsub(/\r/,"",$1); c[$1]++} END {for(k in c) print k,c[k]}' /private/tmp/hERG-MFFGNN-v2/dataset/hERG.csv
ruby -rcsv -e 'ARGV.each{|f| rows=CSV.table(f); puts f; p rows.group_by{|r| r[:used_as]}.transform_values(&:length); p rows.group_by{|r| (r[:pic50].to_f>=5 ? 1:0)}.transform_values(&:length)}' /private/tmp/MultiCTox-v2/MultiCar/data/hERG/data_herg_dev.csv
rg -n 'best_test|test_loader|load_fold_data|torch.save' /private/tmp/hERG-MFFGNN-v2/*.py
```

Important interpretation boundaries:

- Repository row counts describe released artifacts, not necessarily every row described in a paper revision.
- Paper headline metrics remain paper-reported until reproduced on frozen local splits.
- Missing license files are reported as “no license found in audited snapshot,” not as a legal conclusion.
- The hERGBoost 10,798-row count is documented in the official patent publication; the frozen downloadable development file still requires checksum verification before manuscript submission. The 706-row external metrics and tail analysis are primary-reported.
- Local counts have explicit grain: 407,698 admitted **observations**, 369,546 **unique structures**, 343,909 confirmed-WT **observations**, and 63,789 WT-or-unspecified **observations**.
