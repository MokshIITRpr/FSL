"""
Run a sweep over n_shot for few-shot training and testing.

This script launches `train_few_shot.py` multiple times with different
n_shot values (e.g., 1..5). Each run saves checkpoints under a per-shot
checkpoint directory and writes results to `results_fixed/` with clear names.

Example:
    python -u few_shot_sketch_recognition/run_shot_sweep.py \
        --dataset tuberlin \
        --data_root /path/to/TU-Berlin/ \
        --encoder resnet18 \
        --few_shot_model prototypical \
        --distance_metric cosine \
        --feature_norm \
        --logit_scale 20 \
        --pretrained_encoder checkpoints/ssl/simclr_tuberlin_resnet18/best_model.pth \
        --n_way 5 --n_query 15 --epochs 100 \
        --augmentation_strength medium \
        --results_dir results_fixed \
        --train_shots 1,2,3,4,5
"""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description='Run n-shot sweep for few-shot training/testing')
    # Dataset/model core
    p.add_argument('--dataset', type=str, required=True, choices=['tuberlin', 'quickdraw'])
    p.add_argument('--data_root', type=str, required=True)
    p.add_argument('--encoder', type=str, default='sketch_cnn', choices=['sketch_cnn', 'resnet18', 'resnet34', 'resnet50'])
    p.add_argument('--few_shot_model', type=str, default='prototypical', choices=['prototypical', 'matching', 'relation'])
    p.add_argument('--embedding_dim', type=int, default=512)
    p.add_argument('--distance_metric', type=str, default='euclidean', choices=['euclidean', 'cosine'])
    p.add_argument('--pretrained_encoder', type=str, default=None)
    # Training
    p.add_argument('--n_way', type=int, default=5)
    p.add_argument('--n_query', type=int, default=15)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--train_classes', type=int, default=200)
    p.add_argument('--image_size', type=int, default=224)
    p.add_argument('--augmentation_strength', type=str, default='medium', choices=['weak', 'medium', 'strong'])
    # System
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--results_dir', type=str, default='results_fixed')
    p.add_argument('--base_checkpoint_dir', type=str, default='checkpoints/few_shot')
    # Proto improvements
    p.add_argument('--logit_scale', type=float, default=10.0)
    p.add_argument('--feature_norm', action='store_true')
    p.add_argument('--freeze_encoder', action='store_true')
    # Shots
    p.add_argument('--train_shots', type=str, default='1,2,3,4,5', help='Comma-separated list of training shots to run')
    p.add_argument('--eval_shots', type=str, default='1,2,3,4,5', help='Comma-separated list of eval shots per run')
    return p.parse_args()


def main():
    args = parse_args()
    shots = [int(s.strip()) for s in args.train_shots.split(',') if s.strip()]

    train_script = Path(__file__).parent / 'train_few_shot.py'
    py = sys.executable or 'python3'
    print(f"Using python interpreter: {py}")

    for shot in shots:
        exp_ckpt = Path(args.base_checkpoint_dir) / f"{args.few_shot_model}_{args.dataset}_{args.encoder}_{args.distance_metric}_w{args.n_way}_s{shot}_q{args.n_query}"
        exp_ckpt.mkdir(parents=True, exist_ok=True)

        cmd = [
            py, '-u', str(train_script),
            '--dataset', args.dataset,
            '--data_root', args.data_root,
            '--encoder', args.encoder,
            '--few_shot_model', args.few_shot_model,
            '--embedding_dim', str(args.embedding_dim),
            '--distance_metric', args.distance_metric,
            '--n_way', str(args.n_way),
            '--n_shot', str(shot),
            '--n_query', str(args.n_query),
            '--n_train_episodes', '1000',
            '--n_val_episodes', '200',
            '--n_test_episodes', '600',
            '--epochs', str(args.epochs),
            '--lr', str(args.lr),
            '--weight_decay', str(args.weight_decay),
            '--num_workers', str(args.num_workers),
            '--device', args.device,
            '--checkpoint_dir', str(exp_ckpt),
            '--image_size', str(args.image_size),
            '--augmentation_strength', args.augmentation_strength,
            '--results_dir', args.results_dir,
            '--eval_shots', args.eval_shots,
            '--logit_scale', str(args.logit_scale)
        ]

        if args.pretrained_encoder:
            cmd += ['--pretrained_encoder', args.pretrained_encoder]
        if args.feature_norm:
            cmd += ['--feature_norm']
        if args.freeze_encoder:
            cmd += ['--freeze_encoder']

        print('=' * 80)
        print(f"Running {args.n_way}-way {shot}-shot training + eval: {' '.join(cmd)}")
        print('=' * 80)
        subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
