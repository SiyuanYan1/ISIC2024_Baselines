import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import numpy as np
import os
import wandb
import time
import datetime
import warnings
import random
import argparse
from sklearn.metrics import confusion_matrix
import json
from torch.utils.data import WeightedRandomSampler
from metrics import compute_isic_metrics
from dataset import Uni_Dataset
from flash.core.optimizers import LinearWarmupCosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


warnings.simplefilter('ignore')


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Add this line
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Change this to False

    # Add these lines
    # torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


# %% [code] {"jupyter":{"outputs_hidden":false}}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ViT_Net(nn.Module):
    def __init__(self, arch):
        super(ViT_Net, self).__init__()
        self.arch = arch

    def forward(self, x):
        """
        No sigmoid in forward because we are going to use BCEWithLogitsLoss
        Which applies sigmoid for us when calculating a loss
        """

        try:
            out = self.arch(x).logits
        except Exception:
            out = self.arch(x)
        return out


def main(args):
    seed_everything(args.seed)
    # %% [code] {"jupyter":{"outputs_hidden":false}}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.backbone == 'ViT':
        import timm
        arch = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=args.num_class)
    elif args.backbone == 'Dinov2':
        from transformers import Dinov2ForImageClassification
        arch = Dinov2ForImageClassification.from_pretrained('facebook/dinov2-base', num_labels=args.num_class, ignore_mismatched_sizes=True)

    model = ViT_Net(arch=arch)  # New model for each fold
    model = model.to(device)

    epochs = args.epochs  # Number of epochs to run
    es_patience = 6  # Early Stopping patience - for how many epochs with no improvements to wait
    best_val = 0  # Best validation score within this fold
    patience = es_patience  # Current patience counter

    model = ViT_Net(arch=arch)  # New model for each fold
    model = model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = LinearWarmupCosineAnnealingLR(optim, warmup_epochs=3, max_epochs=args.epochs)
    # scheduler = CosineAnnealingLR(optim, T_max=epochs, eta_min=1e-6)

    # mean and std for imagenet
    mean = [0.485, 0.456, 0.406]
    std = [0.228, 0.224, 0.225]

    normalize = transforms.Normalize(mean=mean, std=std)
    input_size = 224
    train_trans = [
        transforms.Resize(224),
        transforms.RandomResizedCrop(input_size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45),
        transforms.ColorJitter(hue=0.2),
        transforms.ToTensor(),
        normalize]

    val_trans = [
        transforms.Resize(224),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize
    ]

    data_transforms = {
        'train': transforms.Compose(train_trans),
        'val': transforms.Compose(val_trans),
        'test': transforms.Compose(val_trans)
    }

    # Load dataset
    df = pd.read_csv(args.csv_file)
    binary = True
    dataset_train = Uni_Dataset(
        df=df,
        root=args.data_dir,
        train=True,
        transforms=data_transforms['train'],
        binary=binary,
        data_percent=1.0
    )
    dataset_val = Uni_Dataset(
        df=df,
        root=args.data_dir,
        val=True,
        transforms=data_transforms['val'],
        binary=binary
    )
    dataset_test = Uni_Dataset(
        df=df,
        root=args.data_dir,
        test=True,
        transforms=data_transforms['test'],
        binary=binary
    )
    print(' => Train size:', len(dataset_train), ',val size:', len(dataset_val), ',test size:', len(dataset_test))

    os.makedirs(args.log_dir, exist_ok=True)
    if args.weights:
        num_one, num_two = dataset_train.count_label()
        weights = [1 / num_one, 1 / num_two]
        print(' => Label distribution:', num_one, num_two)
        train_y = df[(df['split'] == 'train')]['binary_label'].values.tolist()
        sample_weights = np.array([weights[t] for t in train_y])
        sample_weights = torch.from_numpy(sample_weights)
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(dataset_train), replacement=True)
        train_loader = DataLoader(dataset=dataset_train, sampler=sampler, batch_size=args.batch_size,
                                  num_workers=args.num_workers)
    else:
        train_loader = DataLoader(dataset=dataset_train, shuffle=True, batch_size=args.batch_size,
                                  num_workers=args.num_workers)
    val_loader = DataLoader(dataset=dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model_path = args.runs  # Path and filename to save model to

    criterion = nn.CrossEntropyLoss()

    """train and validation"""
    print(' => Start training')
    for epoch in range(epochs):
        start_time = time.time()
        correct = 0
        epoch_loss = 0
        model.train()
        for x, y in tqdm(train_loader):
            x = torch.tensor(x, device=device, dtype=torch.float32)
            y = torch.tensor(y, device=device, dtype=torch.float32)
            y = y.long()
            optim.zero_grad()
            z = model(x)
            loss = criterion(z, y)
            loss.backward()
            optim.step()
            _, pred = torch.max(z, 1)
            correct += (pred.cpu() == y.cpu()).sum().item()
            epoch_loss += loss.item()

        print(" => Evaluation running...")
        val_labels = []
        model.eval()
        val_preds = torch.zeros((len(dataset_val), 2), dtype=torch.float32, device=device)
        val_preds_1d = torch.zeros((len(dataset_val), 1), dtype=torch.float32, device=device)
        with torch.no_grad():  # Do not calculate gradient since we are only predicting
            for j, (x_val, y_val) in enumerate(tqdm(val_loader)):
                x_val = torch.tensor(x_val, device=device, dtype=torch.float32)
                y_val = torch.tensor(y_val, device=device, dtype=torch.float32)
                y_val = y_val.long()
                val_labels += (y_val.tolist())
                z_val = model(x_val)
                val_loss = criterion(z_val, y_val)
                _, val_pred_1d = torch.max(z_val.data, 1)
                val_pred_1d = torch.unsqueeze(val_pred_1d, 1)
                val_preds_1d[j * val_loader.batch_size:j * val_loader.batch_size + x_val.size(0)] = val_pred_1d
                val_preds[j * val_loader.batch_size:j * val_loader.batch_size + x_val.size(0)] = z_val
            val_labels = torch.tensor(val_labels)
            AUC, SEN, SPEC, BACC = compute_isic_metrics(val_labels, val_preds, args.log_dir)
            val_roc = AUC
            print(
                ' => Epoch {:03}: | Train Loss: {:.3f} | Val Loss: {:.3f}|Val roc_auc: {:.6f} | Val Sen: {:.6f} |Val SPEC: {:.6f} |Val bacc: {:.3f} | Training time: {}'.format(
                    epoch + 1,
                    epoch_loss,
                    val_loss,
                    val_roc,
                    SEN,
                    SPEC,
                    BACC,
                    str(datetime.timedelta(seconds=time.time() - start_time))[:7]
                )
            )
            if not args.debug:
                wandb.log({'train loss': epoch_loss, 'val loss': val_loss, 'val auc': val_roc, 'Sen': SEN, 'Spec': SPEC})

            scheduler.step(SEN)
            if SEN >= best_val:
                best_val = SEN
                patience = es_patience  # Resetting patience since we have new best validation accuracy
                print(' => Saving model with best sensitivity: {:.3f}'.format(best_val))
                torch.save(model, os.path.join(args.log_dir, model_path))  # Saving current best model
            else:
                patience -= 1
                if patience == 0:
                    print(' => Early stopping. Best val sensitivity: {:.3f}'.format(best_val))
                    break

    # Model testing
    print(' => Testing')
    model = torch.load(os.path.join(args.log_dir, model_path))  # Loading best model of this fold
    model.eval()  # switch model to the evaluation mode
    TTA = 1
    model.eval()  # switch model to the evaluation mode
    tta_preds = torch.zeros((len(dataset_test), 2), dtype=torch.float32, device=device)
    preds = torch.zeros((len(dataset_test), 2), dtype=torch.float32, device=device)  # Predictions for test test
    test_preds_1d = torch.zeros((len(dataset_test), 1), dtype=torch.float32, device=device)

    for tta in range(TTA):
        test_labels = []
        with torch.no_grad():
            for i, (x_test, y_test) in enumerate(tqdm(test_loader)):
                x_test = torch.tensor(x_test, device=device, dtype=torch.float32)
                y_test = torch.tensor(y_test, device=device, dtype=torch.float32)
                y_test = y_test.long()
                test_labels += (y_test.tolist())
                z_test = model(x_test)
                _, test_pred_1d = torch.max(z_test.data, 1)
                test_pred_1d = torch.unsqueeze(test_pred_1d, 1)
                test_preds_1d[i * test_loader.batch_size:i * test_loader.batch_size + x_test.size(0)] = test_pred_1d
                tta_preds[i * test_loader.batch_size:i * test_loader.batch_size + x_test.size(0)] = z_test
                val_pred = z_test[:, 1]
                val_pred = torch.unsqueeze(val_pred, 1)
                tta_preds[i * test_loader.batch_size:i * test_loader.batch_size + x_test.size(0)] += val_pred

            if tta == 0:
                test_labels = torch.tensor(test_labels)
                labels = test_labels
                test_predictions = {'targets': test_labels, 'preds': test_preds_1d.squeeze(1)}
    preds += tta_preds / TTA

    AUC_test, SEN_test, SPEC_test, BACC_test = compute_isic_metrics(labels, preds, args.log_dir, test=True)

    test_roc = AUC_test
    print(f"RESULTS,{args.seed},{test_roc},{SPEC_test},{SEN_test},{BACC_test}")
    cm = confusion_matrix(labels.cpu().numpy(), test_preds_1d.cpu().numpy())
    print(f"CONFUSION_MATRIX,{cm.ravel()[0]},{cm.ravel()[1]},{cm.ravel()[2]},{cm.ravel()[3]}")

    # Save test predictions
    results = {
        'test_roc': test_roc,
        'test_sen': SEN_test,
        'test_spec': SPEC_test,
        'test_bacc': BACC_test
    }
    with open(os.path.join(args.log_dir, 'results.json'), 'w') as f:
        f.write(json.dumps(results, indent=4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train data with baseline snapddataset model')
    # data
    parser.add_argument('--data-dir', default='data/V3', help='data directory')
    parser.add_argument('--csv-file', default='data/V1/dataset_v1_fold_1.csv', help='path to csv file')
    parser.add_argument('--checkpoints', default='checkpoints', help='where to store checkpoints')
    parser.add_argument('--log-dir', default='output_dir/', help='where to store results')
    parser.add_argument('--device', default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), help='device {cuda:0, cpu}')
    # transformer
    parser.add_argument('--backbone', default='ResNet', help='core cnn model')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
    # training
    parser.add_argument('--valid-as-test', default=False, action='store_true', help='use validation data as test data')
    parser.add_argument('--resume', default=False, action='store_true', help='resume training')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--image-size', type=int, default=224, help='image size to the model')
    parser.add_argument('--num-workers', type=int, default=8, help='number of loader workers')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate 3e-4')
    parser.add_argument('--lr-backbone', type=float, default=3e-4, help='learning rate for backbone')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--lr_drop', type=int, default=50, help='lr drop')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma in StepLR')
    parser.add_argument('--factor', type=float, default=0.1, help='factor in lr_scheduler')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--start-epoch', type=int, default=0, help='start at epoch')
    parser.add_argument('--log-steps', type=int, default=100, help='start at epoch')
    # other
    parser.add_argument('--debug', default=False, action='store_true', help='turn on debug mode')
    parser.add_argument('--debug-count', type=int, default=0, help='# of minibatchs for fast testing, 0 to disable')
    parser.add_argument('--num-class', type=int, default=2, help='# of classes')
    parser.add_argument('--wname', default='baseline_solar_res2', help='pt name')
    parser.add_argument('--runs', default='vit_demo.pth', help='run name')
    parser.add_argument('--weights', default=False, action='store_true', help='h1 or h2 as noisy data?')
    # In your argument parser, add:
    parser.add_argument('--seed', type=int, default=47, help='random seed')

    args = parser.parse_args()
    if not args.debug:
        wandb.init(
            name=args.log_dir,
            project="ISIC2024",
            notes="baselines",
            tags=["isic-demo"],
            config=args
        )
    main(args)
    if not args.debug:
        wandb.finish()
