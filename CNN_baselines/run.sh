CUDA_VISIBLE_DEVICES=0 python base_train.py \
  --batch-size 256 \
  --lr 5e-4 \
  --epochs 20 \
  --backbone ResNet \
  --data-dir ISIC2024/images/ \
  --csv-file ISIC2024_demo.csv \
  --runs model.pth \
  --weights \
  --log-dir output_dir/output_dir_resnet_weight/

CUDA_VISIBLE_DEVICES=0 python base_train.py \
  --batch-size 256 \
  --lr 5e-4 \
  --epochs 20 \
  --backbone ResNet \
  --data-dir ISIC2024/images/ \
  --csv-file ISIC2024_demo.csv \
  --runs model.pth \
  --log-dir output_dir/output_dir_resnet_nonweight/

CUDA_VISIBLE_DEVICES=0 python base_train.py \
  --batch-size 256 \
  --lr 5e-4 \
  --epochs 20 \
  --backbone EfficientNet \
  --data-dir ISIC2024/images/ \
  --csv-file ISIC2024_demo.csv \
  --runs model.pth \
  --weights \
  --log-dir output_dir/output_dir_effnet_weight/


