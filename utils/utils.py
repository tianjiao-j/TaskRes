import torch
import torch.nn.functional as F
import clip
from sklearn.decomposition import FastICA
import os
from datetime import datetime
from tqdm import tqdm
import itertools
from os.path import dirname, exists, join
import numpy as np

torch.manual_seed(1)
temperature = 100.


def get_affinity(cfg, do_ica, features, cache_keys, adapter):
    if do_ica:
        if not cfg['normalize_affinity']:
            features /= features.norm(dim=-1, keepdim=True)
        if cfg['append_linear']:
            affinity = features @ adapter(cache_keys)
        else:
            if adapter:
                affinity = features @ adapter(cache_keys.T).T
            else:
                affinity = features @ cache_keys
        if cfg['normalize_affinity']:
            affinity = normalize_affinity(affinity)
    else:
        if adapter:
            affinity = adapter(features)
        else:
            affinity = features @ cache_keys

    return affinity


def normalize_affinity(affinity):
    x_min = affinity.min(dim=1, keepdim=True)[0]
    x_max = affinity.max(dim=1, keepdim=True)[0]
    affinity = (affinity - x_min) / (x_max - x_min)

    # x_min = affinity.min(dim=1, keepdim=True)[0].min()
    # x_max = affinity.max(dim=1, keepdim=True)[0].max()
    # affinity = (affinity -x_min) / (x_max - x_min)
    return affinity


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def clip_classifier(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights


def build_cache_model(cfg, clip_model, train_loader_cache):
    shots_to_n_components = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4}
    if cfg['shots'] in shots_to_n_components.keys():
        n_components = cfg['n_components'][shots_to_n_components[cfg['shots']]]
    else:
        n_components = 1024
    key_path = join(cfg['cache_dir_shared'], 'keys_' + str(cfg['shots']) + "shots.pt")
    value_path = join(cfg['cache_dir_shared'], 'values_' + str(cfg['shots']) + "shots.pt")
    ica_key_path = join(cfg['cache_dir_shared'], 'ica_keys_' + str(n_components) + '_' + str(cfg['shots']) + "shots.pt")
    ica_value_path = join(cfg['cache_dir_shared'],
                          'ica_values_' + str(n_components) + '_' + str(cfg['shots']) + "shots.pt")
    # fixme
    if cfg['imagenet_ica']:
        if cfg['ica_feature'] == 'imagenet_val':
            print('Using imagenet validation set for ICA')
            imagenet_shot = cfg['imagenet_shot']
            ica_component_path = join(dirname(cfg['cache_dir_shared']), 'imagenet_val',
                                      'ica_component_' + str(n_components) + '_' + str(imagenet_shot) + "shots.pt")
            ica_feature_path = join(dirname(cfg['cache_dir_shared']), 'imagenet_val',
                                    'keys_{}shots.pt'.format(str(imagenet_shot)))
        elif cfg['ica_feature'] == 'imagenet_train':
            print('Using imagenet training set for ICA')
            imagenet_shot = cfg['imagenet_shot']
            ica_component_path = join(dirname(cfg['cache_dir_shared']), 'imagenet',
                                      'ica_component_' + str(n_components) + '_' + str(imagenet_shot) + "shots.pt")
            ica_feature_path = join(dirname(cfg['cache_dir_shared']), 'imagenet',
                                    'keys_{}shots.pt'.format(str(imagenet_shot)))
        elif cfg['ica_feature'] == 'val_all':
            print('Using all validation sets for ICA')
            ica_feature_path = join(dirname(cfg['cache_dir_shared']), 'val_f_all.pt')
            ica_component_path = join(dirname(cfg['cache_dir_shared']),
                                      'ica_component_val_all_{}.pt'.format(n_components))
        else:
            print('Invalid ica_feature')
            exit(1)
    else:
        ica_component_path = join(cfg['cache_dir_shared'],
                                  'ica_component_' + str(n_components) + '_' + str(cfg['shots']) + "shots.pt")

    load_cache = cfg['load_cache']
    has_key_value = exists(key_path) or exists(value_path)
    has_ica_key_value = exists(ica_key_path) or exists(ica_value_path)
    has_ica_component = exists(ica_component_path)

    build_cache = not load_cache or not has_key_value or not has_ica_key_value or not has_ica_component

    # assert not build_cache

    if build_cache:
        print('Building cache model...')
        if not has_key_value:
            print('Creating cache key and values...')
            cache_keys = []
            cache_values = []

            with torch.no_grad():
                # Data augmentation for the cache model
                for augment_idx in range(cfg['augment_epoch']):
                    train_features = []

                    for i, (images, target) in enumerate(train_loader_cache):
                        images = images.cuda()
                        image_features = clip_model.encode_image(images)
                        train_features.append(image_features)
                        if augment_idx == 0:
                            target = target.cuda()
                            cache_values.append(target)
                    cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))

                cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
                cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
                cache_keys = cache_keys.permute(1, 0)

                cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()
        else:
            print('Loading cache key and values...')
            cache_keys = torch.load(key_path)
            cache_values = torch.load(value_path)

        clip_dim = cache_keys.shape[0]

        if not has_ica_key_value or not has_ica_component or not load_cache:
            print('Creating ICA cache key and values...')
            # cache_keys_ica = torch.cat(cache_keys, dim=0).mean(dim=0)
            # cache_keys_ica /= cache_keys_ica.norm(dim=-1, keepdim=True)
            transformer = FastICA(n_components, random_state=cfg['random_state'], whiten=cfg['whiten'],
                                  max_iter=cfg['max_iter'])
            if not cfg['imagenet_ica']:
                print('Fitting ICA with cache data...')
                if n_components > cache_keys.shape[0] or n_components > cache_keys.shape[1]:
                    n_components = min(cache_keys.shape[0], cache_keys.shape[1])
                    print('Reducing ICA components to {}'.format(n_components))
                transformer.fit(cache_keys.T.cpu())
                ica_components = torch.from_numpy(transformer.components_).to(torch.float32).cuda()
                cache_keys_ica = (cache_keys.T.to(torch.float32) @ ica_components.T).T
            else:
                if exists(ica_component_path):
                    print('Loading ICA components from {}'.format(ica_component_path))
                    ica_components = torch.load(ica_component_path).to(torch.float32).cuda()
                else:
                    print('Fitting ICA with selected data, loading features from {}'.format(ica_feature_path))
                    ica_feature = torch.load(ica_feature_path)

                    if ica_feature.shape[1] != clip_dim:
                        ica_feature = ica_feature.T
                    transformer.fit(ica_feature.cpu())
                    ica_components = torch.from_numpy(transformer.components_).to(torch.float32).cuda()

                if ica_components.shape[1] != clip_dim:
                    ica_components = ica_components.T
                cache_keys_ica = (cache_keys.T.to(torch.float32).to('cuda') @ ica_components.T).T

            cache_values_ica = cache_values
        else:
            print('Loading ICA cache key and values...')
            cache_keys_ica = torch.load(ica_key_path)
            cache_values_ica = torch.load(ica_value_path)
            ica_components = torch.load(ica_component_path)

        if not exists(value_path):
            print('Saving cache values to {}'.format(value_path))
            torch.save(cache_values, value_path)
        if not exists(ica_value_path):
            print('Saving ICA values to {}'.format(ica_value_path))
            torch.save(cache_values_ica, ica_value_path)
        if not exists(key_path):
            print('Saving cache keys to {}'.format(key_path))
            torch.save(cache_keys, key_path)
        if not exists(ica_key_path):
            print('Saving ICA keys to {}'.format(ica_key_path))
            torch.save(cache_keys_ica, ica_key_path)
        if not exists(ica_component_path):
            print('Saving ICA components to {}'.format(ica_component_path))
            torch.save(ica_components, ica_component_path)

    else:
        print('Loading cache model...')
        cache_keys = torch.load(key_path)
        cache_values = torch.load(value_path)
        cache_keys_ica = torch.load(ica_key_path)
        cache_values_ica = torch.load(ica_value_path)
        ica_components = torch.load(ica_component_path)

    return cache_keys, cache_values, cache_keys_ica, cache_values_ica, ica_components


def pre_load_features(cfg, split, clip_model, loader):
    # feature_path = join(cfg['cache_dir_shared'], split + "_f.pt")
    # label_path = join(cfg['cache_dir_shared'], split + "_l.pt")

    # if cfg['load_pre_feat'] == False or not exists(feature_path) or not exists(label_path):
    features, labels = [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            images, target = batch['img'], batch['label']
            images, target = images.cuda(), target.cuda()
            image_features = clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            features.append(image_features)
            labels.append(target)

    features, labels = torch.cat(features), torch.cat(labels)

        # torch.save(features, feature_path)
        # torch.save(labels, label_path)

    # else:
    #     features = torch.load(feature_path)
    #     labels = torch.load(label_path)

    return features, labels


def search_hp(f, cfg, cache_keys, cache_keys_ica, cache_values_ica, cache_values, features, features_ica, labels,
              clip_weights, do_ica, adapter=None, adapter_ica=None, zs_classifier=None):
    if cfg['search_hp']:
        with torch.no_grad():
            if cache_values_ica is None:
                cache_values_ica = cache_values

            best_beta, best_alpha = cfg['init_beta'], cfg['init_alpha']
            best_beta_ica, best_alpha_ica = cfg['init_beta_ica'], cfg['init_alpha_ica']
            best_gamma = cfg['init_gamma']

            best_acc = 0.

            if zs_classifier is None:
                clip_logits = temperature * features @ clip_weights
            else:
                if not cfg['add_zs']:
                    print('using zs classifier')
                    clip_logits = temperature * zs_classifier(features)
                else:
                    print('using original CLIP weights')
                    clip_logits_0 = temperature * features @ clip_weights
                    clip_logits = clip_logits_0 + temperature * zs_classifier(features) * best_gamma

            alpha_scale_ica = best_alpha * 2.
            alpha_step_ica = cfg['search_step'][1] * 2

            beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in
                         range(cfg['search_step'][0])]
            alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in
                          range(cfg['search_step'][1])]
            beta_ica_list = beta_list
            # alpha_ica_list = [i * (alpha_scale_ica - 0.1) / alpha_step_ica + 0.1 for i in range(alpha_step_ica)]
            alpha_ica_list = alpha_list
            gamma_list = np.linspace(best_gamma / 10., best_gamma * 10., num=4000).tolist()

            if do_ica:
                paramlist = list(itertools.product(beta_list, alpha_list, beta_ica_list, alpha_ica_list))
                paramlist = list(itertools.product(beta_ica_list, alpha_ica_list))
                beta, alpha = best_beta, best_alpha

                if not cfg['normalize_affinity']:
                    features_ica /= features_ica.norm(dim=-1, keepdim=True)
                if adapter_ica:
                    affinity_ica = features_ica @ adapter_ica(cache_keys_ica.T).T
                else:
                    affinity_ica = features_ica @ cache_keys_ica
                if cfg['normalize_affinity']:  # normalize affinity to [0,1]
                    affinity_ica = normalize_affinity(affinity_ica)
                # affinity_ica = get_affinity(cfg, do_ica, features_ica, cache_keys_ica, adapter_ica)

                for params in tqdm(paramlist):
                    if len(params) == 2:
                        beta_ica, alpha_ica = params
                    elif len(params) == 4:
                        beta, alpha, beta_ica, alpha_ica = params

                    cache_logits_ica = ((-1) * (beta_ica - beta_ica * affinity_ica)).exp() @ cache_values_ica
                    tip_logits = clip_logits + cache_logits_ica * alpha_ica

                    acc = cls_acc(tip_logits, labels)

                    if acc > best_acc:
                        best_acc = acc
                        best_beta_ica = beta_ica
                        best_alpha_ica = alpha_ica
                        best_beta = beta
                        best_alpha = alpha

            else:
                if adapter:
                    affinity = adapter(features)
                else:
                    affinity = features @ cache_keys
                # affinity = get_affinity(cfg, do_ica, features, cache_keys, adapter)
                for beta in beta_list:
                    for alpha in alpha_list:
                        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                        tip_logits = clip_logits + cache_logits * alpha

                        acc = cls_acc(tip_logits, labels)

                        if acc > best_acc:
                            best_acc = acc
                            best_beta = beta
                            best_alpha = alpha

            if cfg['add_zs'] and zs_classifier is not None:
                # if zs_classifier is not None:
                for gamma in tqdm(gamma_list):
                    clip_logits_0 = temperature * features @ clip_weights
                    clip_logits = clip_logits_0 + temperature * zs_classifier(features) * gamma

                    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values
                    tip_logits = clip_logits + cache_logits * best_alpha

                    if do_ica:
                        cache_logits_ica = ((-1) * (
                                best_beta_ica - best_beta_ica * affinity_ica)).exp() @ cache_values_ica
                        tip_logits = tip_logits + cache_logits_ica * best_alpha_ica

                    acc = cls_acc(tip_logits, labels)
                    if acc > best_acc:
                        best_acc = acc
                        best_gamma = gamma

        s = ("After searching, the best accuracy: {:.2f}, beta: {:.2f}, alpha: {:.2f}, beta_ica: {:.2f}, "
             "alpha_ica: {:.2f}, gamma: {:.2f}.").format(best_acc, best_beta, best_alpha, best_beta_ica,
                                                         best_alpha_ica, best_gamma)
        print(s)
        f.write(s + '\n')

    cfg['best_beta'].append(best_beta)
    cfg['best_alpha'].append(best_alpha)
    cfg['best_beta_ica'].append(best_beta_ica)
    cfg['best_alpha_ica'].append(best_alpha_ica)
    cfg['best_gamma'].append(best_gamma)

    return best_beta, best_alpha, best_beta_ica, best_alpha_ica, best_gamma
