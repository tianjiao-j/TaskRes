from os import path, environ
import random
import argparse

import torch
import yaml
import torchvision.transforms as transforms
import torch.nn as nn
from datasets import build_dataset
from datasets.utils import build_data_loader
from utils.utils import *
from datetime import datetime
import socket
from utils.utils import *
from datasets.imagenet_sketch import ImageNetSketch
from datasets.imagenetv2 import ImageNetV2
from datasets.imagenet_a import ImageNetA
from datasets.imagenet_r import ImageNetR


def pre_load_features(clip_model, loader):
    features, labels = [], []
    # class_to_idx = loader.dataset.class_to_idx
    # class_to_idx = {v: k for k, v in class_to_idx.items()}
    idx_to_idx = loader.dataset.idx_to_idx

    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            target = torch.tensor([int(idx_to_idx[int(i)]) for i in target])
            images, target = images.cuda(), target.cuda()
            image_features = clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            features.append(image_features)
            labels.append(target)

    features, labels = torch.cat(features), torch.cat(labels)

    return features, labels


if __name__ == '__main__':
    # os.chdir('..')
    root_dir = '../../Tip-Adapter/data'
    ica_cache_dir = 'caches_imagenet/caches_2024_08_20-22_31/imagenet'
    ica_cache_dir = 'caches/rn50/caches_imagenet_val/imagenet'
    cache_dir = 'caches_imagenet/caches_2024_08_20-19_54/imagenet'
    cache_dir = 'caches_final/no_zs_ft/caches_imagenet/caches/rn50/caches_2024_08_25-20_56_57_128/imagenet'
    dim = 128

    f = open('results_for_paper/ood_{}.txt'.format(dim), 'w')

    for shot in [16]:  # [1, 2, 4, 8, 16]
        print('\nRunning {} shots'.format(shot))

        dataset_names = ['imagenetv2', 'imagenet-sketch', 'imagenet-r', 'imagenet-a']

        # CLIP
        clip_model, preprocess = clip.load('RN50', device='cuda')
        clip_model.eval()

        for dataset_name in dataset_names:
            print('\n' + dataset_name)
            if dataset_name == 'imagenetv2':
                ood_dataset = ImageNetV2(root_dir, preprocess)
            elif dataset_name == 'imagenet-sketch':
                ood_dataset = ImageNetSketch(root_dir, preprocess)
            elif dataset_name == 'imagenet-r':
                ood_dataset = ImageNetR(root_dir, preprocess)
            elif dataset_name == 'imagenet-a':
                ood_dataset = ImageNetA(root_dir, preprocess)

            test_loader = torch.utils.data.DataLoader(ood_dataset.dataset, batch_size=64, num_workers=8, shuffle=False)
            test_features, test_labels = pre_load_features(clip_model, test_loader)
            test_features = test_features.cuda()

            clip_weights = clip_classifier(ood_dataset.classnames, ood_dataset.template, clip_model).cuda()
            test_features = test_features.to(torch.float32)
            clip_weights = clip_weights.to(torch.float32)
            clip_logits = temperature * test_features @ clip_weights
            acc = cls_acc(clip_logits, test_labels)
            print("**** Dataset: {}, Zero-shot CLIP's test accuracy: {:.2f}. ****".format(dataset_name, acc))  # 53.26.
            f.write("**** Dataset: {}, Zero-shot CLIP's test accuracy: {:.2f}. ****".format(dataset_name, acc) + '\n')

            # load hyper-parameters
            cfg = yaml.load(open('configs/imagenet.yaml', 'r'), Loader=yaml.Loader)
            beta, alpha = cfg['init_beta'], cfg['init_alpha']
            beta_ica, alpha_ica = cfg['init_beta_ica'], cfg['init_alpha_ica']
            gamma = cfg['init_gamma']

            # ========================== training-free tip-adapter =============================
            do_ica = False
            cache_keys = torch.load('caches_shared/rn50/imagenet/keys_{}shots.pt'.format(shot)).cuda().to(torch.float32)
            cache_keys_ica = torch.load(
                'caches_shared/rn50/imagenet/ica_keys_{}_{}shots.pt'.format(dim, shot)).cuda().to(
                torch.float32)
            cache_values = torch.load('caches_shared/rn50/imagenet/values_{}shots.pt'.format(shot)).cuda().to(
                torch.float32)

            adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(clip_model.dtype).cuda()
            adapter.weight = nn.Parameter(cache_keys.t())
            adapter_ica = nn.Linear(cache_keys_ica.shape[0], cache_keys_ica.shape[1], bias=False).to(
                clip_model.dtype).cuda()
            adapter_ica.weight = nn.Parameter(cache_keys_ica.t())

            affinity = get_affinity(cfg, do_ica, test_features, cache_keys, adapter)  # adapter = cache_keys
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            tip_logits = clip_logits + cache_logits * alpha

            acc = cls_acc(tip_logits, test_labels)
            print("**** Dataset: {}, Training-free TIP-Adapter's test accuracy: {:.2f}. ****".format(dataset_name, acc))
            f.write("**** Dataset: {}, Training-free TIP-Adapter's test accuracy: {:.2f}. ****".format(dataset_name,
                                                                                                       acc) + '\n')

            # ========================== training-free tip-adapter + ICA =============================
            do_ica = True
            ica_components = torch.load(
                'caches_shared/rn50/imagenet_val/ica_component_{}_50shots.pt'.format(dim)).cuda().to(
                torch.float32)
            test_features_ica = test_features @ ica_components.to('cuda').T

            # affinity = get_affinity(cfg, do_ica, test_features, cache_keys, adapter)  # adapter = cache_keys
            affinity_ica = get_affinity(cfg, do_ica, test_features_ica, cache_keys_ica, adapter_ica)
            # cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            # tip_logits = clip_logits + cache_logits * alpha
            cache_logits = ((-1) * (beta_ica - beta_ica * affinity_ica)).exp() @ cache_values
            tip_logits = clip_logits + cache_logits * alpha_ica

            acc = cls_acc(tip_logits, test_labels)
            print("**** Dataset: {}, Training-free TIP-Adapter + ICA test accuracy: {:.2f}. ****".format(dataset_name,
                                                                                                         acc))
            f.write("**** Dataset: {}, Training-free TIP-Adapter + ICA test accuracy: {:.2f}. ****".format(dataset_name,
                                                                                                           acc) + '\n')

            # =============================== tip-adapter-f =====================================
            do_ica = False
            cache_keys = torch.load(path.join(cache_dir, 'no_ica/best_adapter_{}shots.pt'.format(shot))).cuda()
            adapter.weight = nn.Parameter(cache_keys).cuda()
            in_features, out_features = clip_weights.shape
            zs_classifier = nn.Linear(in_features=in_features, out_features=out_features, bias=False).to(
                torch.float32).cuda()
            # zs_classifier.weight = nn.Parameter(
            #     torch.load(path.join(cache_dir, 'no_ica/best_zs_classifier_{}shots.pt'.format(shot))).cuda())
            #
            # clip_logits = temperature * zs_classifier(test_features)
            affinity = get_affinity(cfg, do_ica, test_features, cache_keys, adapter)  # adapter = cache_keys
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            tip_logits = clip_logits + cache_logits * alpha

            acc = cls_acc(tip_logits, test_labels)
            print("**** Dataset: {}, TIP-Adapter-F's test accuracy: {:.2f}. ****".format(dataset_name, acc))
            f.write("**** Dataset: {}, TIP-Adapter-F's test accuracy: {:.2f}. ****".format(dataset_name, acc) + '\n')

            # =============================== tip-adapter-f + ICA =====================================
            do_ica = True
            # cache_keys = torch.load(path.join(ica_cache_dir, 'ica/best_adapter_{}shots.pt'.format(shot))).cuda()
            # adapter.weight = nn.Parameter(cache_keys)
            ica_cache_dir = 'caches_final/no_zs_ft/caches_imagenet/caches/rn50/caches_2024_08_25-20_56_57_128/imagenet'
            cache_keys_ica = torch.load(path.join(ica_cache_dir, 'ica/best_adapter_ica_{}shots.pt'.format(shot))).cuda()
            adapter_ica = nn.Linear(cache_keys_ica.shape[0], cache_keys_ica.shape[1], bias=False)
            adapter_ica.weight = nn.Parameter(cache_keys_ica)
            # zs_classifier.weight = nn.Parameter(
            #     torch.load(path.join(ica_cache_dir, 'ica/best_zs_classifier_{}shots.pt'.format(shot))))
            #
            # clip_logits = temperature * zs_classifier(test_features)
            # affinity = get_affinity(cfg, do_ica, test_features, cache_keys, adapter)  # fixme: do_ica
            affinity_ica = get_affinity(cfg, do_ica, test_features_ica, cache_keys_ica, adapter_ica)
            # cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            # tip_logits = clip_logits + cache_logits * alpha
            cache_logits = ((-1) * (beta_ica - beta_ica * affinity_ica)).exp() @ cache_values
            tip_logits = clip_logits + cache_logits * alpha_ica

            acc = cls_acc(tip_logits, test_labels)
            print("**** Dataset: {}, TIP-Adapter-F + ICA test accuracy: {:.2f}. ****".format(dataset_name, acc))
            f.write(
                "**** Dataset: {}, TIP-Adapter-F + ICA test accuracy: {:.2f}. ****".format(dataset_name, acc) + '\n')

            # =============================== tip-adapter-f + ICA + ZS-FT =====================================
            do_ica = True
            # cache_keys = torch.load(path.join(ica_cache_dir, 'ica/best_adapter_{}shots.pt'.format(shot))).cuda()
            # adapter.weight = nn.Parameter(cache_keys)
            ica_cache_dir = 'caches_final/zs_ft/caches_imagenet/caches/rn50/caches_2024_08_24-22_43_28_128/imagenet'
            cache_keys_ica = torch.load(path.join(ica_cache_dir, 'ica/best_adapter_ica_{}shots.pt'.format(shot))).cuda()
            adapter_ica = nn.Linear(cache_keys_ica.shape[0], cache_keys_ica.shape[1], bias=False)
            adapter_ica.weight = nn.Parameter(cache_keys_ica)
            zs_classifier.weight = nn.Parameter(
                torch.load(path.join(ica_cache_dir, 'ica/best_zs_classifier_{}shots.pt'.format(shot))))

            clip_logits = temperature * zs_classifier(test_features)
            # affinity = get_affinity(cfg, do_ica, test_features, cache_keys, adapter)  # fixme: do_ica
            affinity_ica = get_affinity(cfg, do_ica, test_features_ica, cache_keys_ica, adapter_ica)
            # cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            # tip_logits = clip_logits + cache_logits * alpha
            cache_logits = ((-1) * (beta_ica - beta_ica * affinity_ica)).exp() @ cache_values
            tip_logits = clip_logits + cache_logits * alpha_ica

            acc = cls_acc(tip_logits, test_labels)
            print("**** Dataset: {}, TIP-Adapter-F + ICA + ZS-FT test accuracy: {:.2f}. ****".format(dataset_name, acc))
            f.write("**** Dataset: {}, TIP-Adapter-F + ICA + ZS-FT test accuracy: {:.2f}. ****".format(dataset_name,
                                                                                                       acc) + '\n')

    f.close()
