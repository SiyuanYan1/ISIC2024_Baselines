# ISIC2024_Baselines

This repo is for a quick start for the ISIC2024 challenge. Notice the code is only for demonstration purpose. We evalaute representative baselines on a sample of the whole datasets, please refer to ISIC2024_demo.csv Statistics section for details.
## Dataset Statistics

- **Total samples:** 401,059
- **Malignant samples:** 393
- **Benign samples:** 400,666

## Distribution by Attribution

| Attribution | Count |
|-------------|-------|
| Memorial Sloan Kettering Cancer Center | 129,068 |
| Department of Dermatology, Hospital Clínic de Barcelona | 105,724 |
| University Hospital of Basel | 65,218 |
| Frazer Institute, The University of Queensland, Dermatology Research Centre | 51,768 |
| ACEMID MIA | 28,665 |
| ViDIR Group, Department of Dermatology, Medical University of Vienna | 12,640 |
| Department of Dermatology, University of Athens, Andreas Syggros Hospital of Skin and Venereal Diseases | 7,976 |

## Distribution of Binary Labels for Each Attribution

| Attribution | Benign | Malignant |
|-------------|--------|-----------|
| ACEMID MIA | 28,632 | 33 |
| Department of Dermatology, Hospital Clínic de Barcelona | 105,652 | 72 |
| Department of Dermatology, University of Athens | 7,970 | 6 |
| Frazer Institute, The University of Queensland | 51,687 | 81 |
| Memorial Sloan Kettering Cancer Center | 128,894 | 174 |
| University Hospital of Basel | 65,205 | 13 |
| ViDIR Group, Department of Dermatology, Medical University of Vienna | 12,626 | 14 |

## ISIC2024_demo.csv Statistics

- **Total samples:** 49,025
- **Test samples:** 28,665
- **Train samples:** 16,288
- **Validation samples:** 4,072

### Distribution by Split

| Split | Count |
|-------|-------|
| Test | 28,665 |
| Train | 16,288 |
| Validation | 4,072 |

### Distribution of Binary Labels for Each Split

| Split | Benign | Malignant |
|-------|--------|-----------|
| Test | 28,632 | 33 |
| Train | 16,000 | 288 |
| Validation | 4,000 | 72 |

### Percentage of Malignant Samples in Each Split

- Test: 0.12%
- Train: 1.77%
- Validation: 1.77%

## Distribution by Attribution in the Demo Dataset

| Attribution | Count |
|-------------|-------|
| ACEMID MIA | 28,665 |
| Memorial Sloan Kettering Cancer Center | 6,979 |
| Department of Dermatology, Hospital Clínic de Barcelona | 5,789 |
| University Hospital of Basel | 3,633 |
| Frazer Institute, The University of Queensland, Dermatology Research Centre | 2,838 |
| ViDIR Group, Department of Dermatology, Medical University of Vienna | 682 |
| Department of Dermatology, University of Athens | 439 |

## Distribution of Binary Labels for Each Attribution in the Demo Dataset

| Attribution | Benign | Malignant |
|-------------|--------|-----------|
| ACEMID MIA | 28,632 | 33 |
| Department of Dermatology, Hospital Clínic de Barcelona | 5,717 | 72 |
| Department of Dermatology, University of Athens | 433 | 6 |
| Frazer Institute, The University of Queensland | 2,757 | 81 |
| Memorial Sloan Kettering Cancer Center | 6,805 | 174 |
| University Hospital of Basel | 3,620 | 13 |
| ViDIR Group, Department of Dermatology, Medical University of Vienna | 668 | 14 |

## Evaluating baselines on ISIC2024_demo.csv
test result using resnet50_non_weightedsample
[[28615    17]
[   33     0]]
Test roc_auc: 0.861951| Spec : 0.999 | SEN : 0.000|Test bacc: 0.500

test result using resnet50_weightedsample
[[25969  2663]
 [   13    20]]
Test roc_auc: 0.855072| Spec : 0.907 | SEN : 0.606

test result using effnet_weightedsample
[[27237  1395]
 [   18    15]]
 Test roc_auc: 0.821604| Spec : 0.951 | SEN : 0.455


