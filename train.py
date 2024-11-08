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
import torch.nn as nn
import yaml
from torchvision.ops import MLP


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
    dataset_name = 'imagenet_a'
    tip_adapter_path = '../Tip_Adapter'
    args.root = '/home/tianjiao/PycharmProjects/Tip-Adapter/data'
    args.seed = 1
    args.trainer = 'TaskRes'
    args.dataset_config_file = '/home/tianjiao/PycharmProjects/TaskRes/configs/datasets/{}.yaml'.format(dataset_name)
    args.config_file = '/home/tianjiao/PycharmProjects/TaskRes/configs/trainers/TaskRes/generalization_rn50.yaml'
    args.outputs_dir = '/home/tianjiao/PycharmProjects/TaskRes/eval_outputs/seed{}'.format(args.seed)
    args.model_dir = '/home/tianjiao/PycharmProjects/TaskRes/output/FINAL/debug/imagenet/adam_lr2e-4_B256_ep200_16shots/seed{}'.format(
        args.seed)
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

    cfg_ood = yaml.load(open(join(tip_adapter_path, 'configs/imagenet.yaml'), 'r'), Loader=yaml.Loader)
    f = open(os.path.join('ood', 'ood_results_{}.txt'.format(dataset_name)), 'w')
    temperature = 100.
    # =========================== Zero-shot CLIP ====================================
    clip_model, preprocess = clip.load('RN50', device='cuda')
    clip_model.eval()
    test_features, test_labels = pre_load_features(cfg_ood, 'test', clip_model, trainer.dm.test_loader)
    test_features = test_features.cpu()
    test_labels = test_labels.cpu()

    clip_weights = clip_classifier(trainer.dm.dataset.classnames, imagenet_templates, clip_model).cpu()
    test_features = test_features.to(torch.float32)
    clip_weights = clip_weights.to(torch.float32).cpu()
    clip_logits = temperature * test_features @ clip_weights
    acc = cls_acc(clip_logits, test_labels)
    print("**** Dataset: {}, Zero-shot CLIP's test accuracy: {:.2f}. ****".format(dataset_name, acc))  # 53.26.
    f.write("**** Dataset: {}, Zero-shot CLIP's test accuracy: {:.2f}. ****".format(dataset_name, acc) + '\n')

    # load hyper-parameters
    beta, alpha = cfg_ood['init_beta'], cfg_ood['init_alpha']
    beta_ica, alpha_ica = cfg_ood['init_beta_ica'], cfg_ood['init_alpha_ica']
    gamma = cfg_ood['init_gamma']
    eta = cfg_ood['init_eta']
    shot = 16
    dim = 1024

    # ========================== training-free tip-adapter =============================
    do_ica = False
    cache_keys = torch.load(
        join(tip_adapter_path, 'caches_shared/rn50/imagenet/keys_{}shots.pt'.format(shot))).cpu().to(torch.float32)

    cache_values = torch.load(
        join(tip_adapter_path, 'caches_shared/rn50/imagenet/values_{}shots.pt'.format(shot))).cpu().to(
        torch.float32)

    adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(clip_model.dtype).cpu()
    adapter.weight = nn.Parameter(cache_keys.t())

    affinity = get_affinity(cfg_ood, do_ica, test_features, cache_keys, adapter)  # adapter = cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    tip_logits = clip_logits + cache_logits * alpha

    acc = cls_acc(tip_logits, test_labels)
    print("**** Dataset: {}, Training-free TIP-Adapter's test accuracy: {:.2f}. ****".format(dataset_name, acc))
    f.write("**** Dataset: {}, Training-free TIP-Adapter's test accuracy: {:.2f}. ****".format(dataset_name,
                                                                                               acc) + '\n')

    # ========================== training-free tip-adapter + ICA =============================
    do_ica = True
    ica_components = torch.load(join(tip_adapter_path,
                                     'caches_shared/rn50/imagenet/ica_component_{}_100shots.pt'.format(dim))).cpu().to(
        torch.float32)
    cache_keys_ica = ((cache_keys.T @ ica_components.T).T).cpu()
    test_features_ica = (test_features @ ica_components.T).cpu()

    # affinity = get_affinity(cfg_ood, do_ica, test_features, cache_keys, adapter)  # adapter = cache_keys
    affinity_ica = get_affinity(cfg_ood, do_ica, test_features_ica, cache_keys_ica, adapter=None)
    # cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    # tip_logits = clip_logits + cache_logits * alpha
    cache_logits = ((-1) * (beta_ica - beta_ica * affinity_ica)).exp() @ cache_values

    A_weight = test_features @ clip_weights * cfg_ood['temp']
    A_weight_v = F.softmax(A_weight, dim=0)
    A_weight_t = F.softmax(A_weight, dim=1)

    feat_t_a = test_features.T @ A_weight_v  # text-guided cache keys
    feat_v_a = A_weight_t @ clip_weights.T  # visual-guided
    # feat_v_a = feat_v_a.mean(0) + feat_v_a.max(0)[0]

    clip_logits_1 = temperature * test_features @ feat_t_a * gamma
    clip_logits_2 = temperature * feat_v_a @ clip_weights * eta
    # clip_logits = clip_logits_0 + clip_logits_1 + clip_logits_2

    tip_logits = clip_logits + cache_logits * alpha_ica + clip_logits_1 + clip_logits_2

    acc = cls_acc(tip_logits, test_labels)
    print("**** Dataset: {}, Training-free TIP-Adapter + ICA test accuracy: {:.2f}. ****".format(dataset_name,
                                                                                                 acc))
    f.write("**** Dataset: {}, Training-free TIP-Adapter + ICA test accuracy: {:.2f}. ****".format(dataset_name,
                                                                                                   acc) + '\n')

    # =============================== tip-adapter-f =====================================
    do_ica = False
    cache_dir = 'caches_final/no_zs_ft/caches_imagenet/caches/rn50/caches_2024_08_25-20_56_57_128/imagenet'
    cache_keys = torch.load(join(tip_adapter_path, cache_dir, 'no_ica/best_adapter_{}shots.pt'.format(shot))).cpu()
    adapter.weight = nn.Parameter(cache_keys).cpu()
    in_features, out_features = clip_weights.shape
    zs_classifier = nn.Linear(in_features=in_features, out_features=out_features, bias=False).to(
        torch.float32).cpu()
    # zs_classifier.weight = nn.Parameter(
    #     torch.load(path.join(cache_dir, 'no_ica/best_zs_classifier_{}shots.pt'.format(shot))).cuda())
    #
    # clip_logits = temperature * zs_classifier(test_features)
    affinity = get_affinity(cfg_ood, do_ica, test_features, cache_keys, adapter)  # adapter = cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    tip_logits = clip_logits + cache_logits * alpha

    acc = cls_acc(tip_logits, test_labels)
    print("**** Dataset: {}, TIP-Adapter-F's test accuracy: {:.2f}. ****".format(dataset_name, acc))
    f.write("**** Dataset: {}, TIP-Adapter-F's test accuracy: {:.2f}. ****".format(dataset_name, acc) + '\n')

    # =============================== tip-adapter-f + ICA + ZS-FT =====================================
    do_ica = True
    # cache_keys = torch.load(path.join(ica_cache_dir, 'ica/best_adapter_{}shots.pt'.format(shot))).cuda()
    # adapter.weight = nn.Parameter(cache_keys)
    ica_cache_dir = 'caches_cvpr/rn50/caches_final/imagenet'
    adapter_ica_weight = join(tip_adapter_path, ica_cache_dir, 'ica/best_adapter_ica_{}shots.pt'.format(shot))
    dim = ica_components.shape[0]
    adapter_ica = MLP(dim, [dim], activation_layer=None, bias=False, dropout=0.0)
    adapter_ica = adapter_ica.to('cpu')
    adapter_ica.load_state_dict(torch.load(adapter_ica_weight))
    zs_classifier.weight = nn.Parameter(
        torch.load(join(tip_adapter_path, ica_cache_dir, 'ica/best_zs_classifier_{}shots.pt'.format(shot))).cpu())

    clip_logits = temperature * zs_classifier(test_features)

    A_weight = test_features @ clip_weights * cfg_ood['temp']
    A_weight_v = F.softmax(A_weight, dim=0)
    A_weight_t = F.softmax(A_weight, dim=1)

    feat_t_a = test_features.T @ A_weight_v  # text-guided cache keys
    feat_v_a = A_weight_t @ clip_weights.T  # visual-guided
    # feat_v_a = feat_v_a.mean(0) + feat_v_a.max(0)[0]

    clip_logits_1 = temperature * test_features @ feat_t_a * gamma
    clip_logits_2 = temperature * feat_v_a @ clip_weights * eta

    # affinity = get_affinity(cfg_ood, do_ica, test_features, cache_keys, adapter)  # fixme: do_ica
    affinity_ica = get_affinity(cfg_ood, do_ica, test_features_ica, cache_keys_ica, adapter_ica)
    # cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    # tip_logits = clip_logits + cache_logits * alpha
    cache_logits = ((-1) * (beta_ica - beta_ica * affinity_ica)).exp() @ cache_values
    tip_logits = clip_logits + cache_logits * alpha_ica + clip_logits_1 + clip_logits_2

    acc = cls_acc(tip_logits, test_labels)
    print("**** Dataset: {}, TIP-Adapter-F + ICA + ZS-FT test accuracy: {:.2f}. ****".format(dataset_name, acc))
    f.write("**** Dataset: {}, TIP-Adapter-F + ICA + ZS-FT test accuracy: {:.2f}. ****".format(dataset_name,
                                                                                               acc) + '\n')

    f.close()

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
    )  # "none" means without using enhanced base
    args = parser.parse_args()
    main(args)
