# ISIC2024_Baselines

Dataset statistic
Total samples: 401059
Malignant samples: 393
Benign samples: 400666

Distribution by attribution:
attribution
Memorial Sloan Kettering Cancer Center                                                                                                                 129068
Department of Dermatology, Hospital Clínic de Barcelona                                                                                                105724
University Hospital of Basel                                                                                                                            65218
Frazer Institute, The University of Queensland, Dermatology Research Centre                                                                             51768
ACEMID MIA                                                                                                                                              28665
ViDIR Group, Department of Dermatology, Medical University of Vienna                                                                                    12640
Department of Dermatology, University of Athens, Andreas Syggros Hospital of Skin and Venereal Diseases, Alexander Stratigos, Konstantinos Liopyris      7976
Name: count, dtype: int64

Distribution of binary_label for each attribution:
binary_label                                             0    1
attribution                                                    
ACEMID MIA                                           28632   33
Department of Dermatology, Hospital Clínic de B...  105652   72
Department of Dermatology, University of Athens...    7970    6
Frazer Institute, The University of Queensland,...   51687   81
Memorial Sloan Kettering Cancer Center              128894  174
University Hospital of Basel                         65205   13
ViDIR Group, Department of Dermatology, Medical...   12626   14

ISIC2024_demo.csv statistics:
Total samples: 49025
Test samples: 28665
Train samples: 16288
Validation samples: 4072

Distribution by split:
split
test     28665
train    16288
val       4072
Name: count, dtype: int64

Distribution of binary_label for each split:
binary_label      0    1
split                   
test          28632   33
train         16000  288
val            4000   72

Percentage of malignant samples in each split:
test: 0.12%
train: 1.77%
val: 1.77%

Distribution by attribution in the new dataset:
attribution
ACEMID MIA                                                                                                                                             28665
Memorial Sloan Kettering Cancer Center                                                                                                                  6979
Department of Dermatology, Hospital Clínic de Barcelona                                                                                                 5789
University Hospital of Basel                                                                                                                            3633
Frazer Institute, The University of Queensland, Dermatology Research Centre                                                                             2838
ViDIR Group, Department of Dermatology, Medical University of Vienna                                                                                     682
Department of Dermatology, University of Athens, Andreas Syggros Hospital of Skin and Venereal Diseases, Alexander Stratigos, Konstantinos Liopyris      439
Name: count, dtype: int64

Distribution of binary_label for each attribution in the new dataset:
binary_label                                            0    1
attribution                                                   
ACEMID MIA                                          28632   33
Department of Dermatology, Hospital Clínic de B...   5717   72
Department of Dermatology, University of Athens...    433    6
Frazer Institute, The University of Queensland,...   2757   81
Memorial Sloan Kettering Cancer Center               6805  174
University Hospital of Basel                         3620   13
ViDIR Group, Department of Dermatology, Medical...    668   14
