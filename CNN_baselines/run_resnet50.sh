#!/bin/bash

# Set up result directory
result_dir="output_dir/resnet_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$result_dir"

seeds=(0 1 10)
output_file="$result_dir/individual_results.csv"
cm_file="$result_dir/individual_confusion_matrices.csv"

# Create header for results file
echo "Seed,AUC,SPEC,SEN,BACC" > $output_file
echo "Seed,TN,FP,FN,TP" > $cm_file

for seed in "${seeds[@]}"
do
    echo "Running experiment with seed $seed"
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --batch-size 256 \
        --lr 5e-4 \
        --epochs 20 \
        --backbone ResNet \
        --data-dir /home/share/Uni_Eval/ISIC2024/images/ \
        --csv-file ISIC2024_demo.csv \
        --runs model_seed${seed}.pth \
        --weights \
        --log-dir $result_dir/output_dir_resnet_weight_seed${seed}/ \
        --seed $seed | tee "$result_dir/run_seed${seed}.log" \
                           >(grep "RESULTS" | cut -d',' -f2- >> $output_file) \
                           >(grep "CONFUSION_MATRIX" | cut -d',' -f2- | sed "s/^/${seed},/" >> $cm_file)
done

# Calculate mean and standard deviation, and average confusion matrix
echo "Calculating mean, standard deviation, and average confusion matrix..."
python - <<EOF
import pandas as pd
import numpy as np

# Process results
df = pd.read_csv('$output_file')
mean = df.mean()
std = df.std()

print("\nResults:")
print(df)
print("\nMean:")
print(mean)
print("\nStandard Deviation:")
print(std)

# Save aggregate results
aggregate = pd.DataFrame({'Metric': mean.index, 'Mean': mean.values, 'Std': std.values})
aggregate.to_csv('$result_dir/aggregate_results.csv', index=False)

# Process confusion matrices
cm_df = pd.read_csv('$cm_file')
avg_cm = cm_df.iloc[:, 1:].mean().round().astype(int)

print("\nAverage Confusion Matrix:")
print(f"TN: {avg_cm['TN']}, FP: {avg_cm['FP']}")
print(f"FN: {avg_cm['FN']}, TP: {avg_cm['TP']}")

# Save average confusion matrix
avg_cm.to_csv('$result_dir/average_confusion_matrix.csv', header=True)

EOF

echo "Results organized in: $result_dir"
echo "Individual run results saved in: $output_file"
echo "Individual confusion matrices saved in: $cm_file"
echo "Aggregate results saved in: $result_dir/aggregate_results.csv"
echo "Average confusion matrix saved in: $result_dir/average_confusion_matrix.csv"