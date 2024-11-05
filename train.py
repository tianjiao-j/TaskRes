import argparse
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# custom
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet

import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r

import trainers.taskres
import trainers.zsclip

from utils.utils import *
from datasets.imagenet import imagenet_templates


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head



def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.TaskRes = CN()
    cfg.TRAINER.TaskRes.N_CTX = 16  # number of context vectors
    cfg.TRAINER.TaskRes.CSC = False  # class-specific context
    cfg.TRAINER.TaskRes.CTX_INIT = ""  # initialization words
    cfg.TRAINER.TaskRes.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.TaskRes.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.TaskRes.RESIDUAL_SCALE = 1.0
    cfg.TRAINER.TaskRes.ENHANCED_BASE = args.enhanced_base

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    args.root = '/home/tianjiao/PycharmProjects/Tip-Adapter/data'
    args.seed = 1
    args.trainer = 'TaskRes'
    args.dataset_config_file = '/home/tianjiao/PycharmProjects/TaskRes/configs/datasets/imagenet_sketch.yaml'
    args.config_file = '/home/tianjiao/PycharmProjects/TaskRes/configs/trainers/TaskRes/generalization_rn50.yaml'
    args.outputs_dir = '/home/tianjiao/PycharmProjects/TaskRes/eval_outputs/seed{}'.format(args.seed)
    args.model_dir = '/home/tianjiao/PycharmProjects/TaskRes/output/FINAL/debug/imagenet/adam_lr2e-4_B256_ep200_16shots/seed{}'.format(args.seed)
    args.load_epoch = 200
    args.eval_only = True

    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    # print("Collecting env info ...")
    # print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    clip_model, preprocess = clip.load('RN50', device='cuda')
    clip_model.eval()
    test_features, test_labels = pre_load_features(None, 'test', clip_model, trainer.dm.test_loader)
    test_features = test_features.cuda()

    clip_weights = clip_classifier(trainer.dm.dataset.classnames, imagenet_templates, clip_model).cuda()
    test_features = test_features.to(torch.float32)
    clip_weights = clip_weights.to(torch.float32)
    clip_logits = temperature * test_features @ clip_weights
    acc = cls_acc(clip_logits, test_labels)
    # print("**** Dataset: {}, Zero-shot CLIP's test accuracy: {:.2f}. ****".format(dataset_name, acc))  # 53.26.
    # f.write("**** Dataset: {}, Zero-shot CLIP's test accuracy: {:.2f}. ****".format(dataset_name, acc) + '\n')

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    parser.add_argument(
        "--enhanced-base", type=str, default="none", help="path to enhanced base classifier weight"
    )   # "none" means without using enhanced base
    args = parser.parse_args()
    main(args)
