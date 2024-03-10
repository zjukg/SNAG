import torch
import random
import json
import numpy as np
import pdb
import torch.distributed as dist
import os
import os.path as osp
from collections import Counter
import pickle
import torch.nn.functional as F
from transformers import BertTokenizer
import torch.distributed
from tqdm import tqdm

from .utils import get_topk_indices, get_adjr
from .data import EADataset

def load_msnea_data(logger, args):
    
    # assert args.data_choice in ["DWY", "DBP15K", "FBYG15K", "FBDB15K"]
    # if args.data_choice in ["DWY", "DBP15K", "FBYG15K", "FBDB15K"]:
    KGs, non_train, train_ill, test_ill, eval_ill, test_ill_ = load_eva_data(logger, args)

    # elif args.data_choice in ["FBYG15K_attr", "FBDB15K_attr"]:
    #     pass

    return KGs, non_train, train_ill, test_ill, eval_ill, test_ill_

def load_eva_data(logger, args):
    if "OEA" in args.data_choice:
        file_dir = osp.join(args.data_path, "OpenEA", args.data_choice)
    else:
        file_dir = osp.join(args.data_path, args.data_choice, args.data_split)
    lang_list = [1, 2]
    msnea_need, ent2id_dict, ills, triples, r_hs, r_ts, ids, rt_dict1, hr_dict1, rt_dict2, hr_dict2, id_relation_triples1, id_relation_triples2 = read_raw_data(file_dir, lang_list)
    e1 = os.path.join(file_dir, 'ent_ids_1')
    e2 = os.path.join(file_dir, 'ent_ids_2')
    
    left_ents = get_ids(e1)
    
    right_ents = get_ids(e2)
    # 39654 = 19661 + 19993
    ENT_NUM = len(ent2id_dict)
    # 2111
    REL_NUM = len(r_hs)
    
    np.random.shuffle(ills)
    
    data_prefix = ""
    if args.ratio != "1.0":
        data_prefix = f"_{args.ratio}"
    if "V1" in file_dir:
        args.data_split = "norm"
        img_vec_path = osp.join(args.data_path, f"OpenEA/pkl/{args.data_choice}_id_img_feature_dict{data_prefix}.pkl")
    elif "V2" in file_dir:
        args.data_split = "dense"
        img_vec_path = osp.join(args.data_path, f"OpenEA/pkl/{args.data_choice}_id_img_feature_dict{data_prefix}.pkl")
    elif "FB" in file_dir:
        img_vec_path = osp.join(args.data_path, f"pkls/{args.data_choice}_id_img_feature_dict{data_prefix}.pkl")
    else:
        # fr_en
        split = file_dir.split("/")[-1]
        # /home/chenzhuo/data/eva/pkls/fr_en_GA_id_img_feature_dict.pkl
        img_vec_path = osp.join(args.data_path, "pkls", args.data_split + f"_GA_id_img_feature_dict{data_prefix}.pkl")

    assert osp.exists(img_vec_path)
    img_features, ent_wo_img, ent_w_img = load_img(logger, ENT_NUM, img_vec_path, ills)
    logger.info(f"image feature shape:{img_features.shape}")

    # MCLEA
    if args.word_embedding == "glove":
        word2vec_path = os.path.join(args.data_path, "embedding", "glove.6B.300d.txt")
    elif args.word_embedding == 'bert':
        pass
    else:
        raise Exception("error word embedding")

    name_features = None
    char_features = None
    if args.data_choice == "DBP15K" and (args.w_name or args.w_char):

        assert osp.exists(word2vec_path)
        ent_vec, char_features = load_word_char_features(ENT_NUM, word2vec_path, args, logger)
        name_features = F.normalize(torch.Tensor(ent_vec))
        char_features = F.normalize(torch.Tensor(char_features))
        logger.info(f"name feature shape:{name_features.shape}")
        logger.info(f"char feature shape:{char_features.shape}")

    if args.unsup:
        mode = args.unsup_mode
        if mode == "char":
            input_features = char_features
        elif mode == "name":
            input_features = name_features
        else:
            input_features = F.normalize(torch.Tensor(img_features))

        train_ill = visual_pivot_induction(args, left_ents, right_ents, input_features, ills, logger)
    else:
        train_ill = np.array(ills[:int(len(ills) // 1 * args.data_rate)], dtype=np.int32)

    test_ill_ = ills[int(len(ills) // 1 * args.data_rate):]
    test_ill = np.array(test_ill_, dtype=np.int32)

    
    test_left = torch.LongTensor(test_ill[:, 0].squeeze())
    test_right = torch.LongTensor(test_ill[:, 1].squeeze())

    
    left_non_train = list(set(left_ents) - set(train_ill[:, 0].tolist()))

    
    right_non_train = list(set(right_ents) - set(train_ill[:, 1].tolist()))

    
    sup_triples1_set, sup_triples2_set = generate_sup_relation_triples(train_ill, rt_dict1, hr_dict1, rt_dict2, hr_dict2)
    # pdb.set_trace()
    relation_triples_list1, relation_triples_list2 = add_sup_relation_triples(sup_triples1_set, sup_triples2_set, id_relation_triples1, id_relation_triples2)
    
    relation_triples_set1 = set(relation_triples_list1)
    relation_triples_set2 = set(relation_triples_list2)
    relation_triples_set = relation_triples_set1 | relation_triples_set2

    new_dict = {"relation_triples_list1": relation_triples_list1,
                "relation_triples_list2": relation_triples_list2,
                "relation_triples_set1": relation_triples_set1,
                "relation_triples_set2": relation_triples_set2,
                "relation_triples_set": relation_triples_set
                }
    msnea_need.update(new_dict)

    # left_non_train = test_ill[:,0].tolist()
    # right_non_train = test_ill[:,1].tolist()
    logger.info(f"#left entity : {len(left_ents)}, #right entity: {len(right_ents)}")
    logger.info(f"#left entity not in train set: {len(left_non_train)}, #right entity not in train set: {len(right_non_train)}")

    # (39654, 1000)
    

    rel_features = load_relation(ENT_NUM, triples, 1000)
    logger.info(f"relation feature shape:{rel_features.shape}")
    
    
    
    a1 = os.path.join(file_dir, 'training_attrs_1')
    a2 = os.path.join(file_dir, 'training_attrs_2')
    att_features = load_attr([a1, a2], ENT_NUM, ent2id_dict, 1000)  # attr
    logger.info(f"attribute feature shape:{att_features.shape}")

    logger.info("-----dataset summary-----")
    logger.info(f"dataset:\t\t {file_dir}")
    logger.info(f"triple num:\t {len(triples)}")
    logger.info(f"entity num:\t {ENT_NUM}")
    logger.info(f"relation num:\t {REL_NUM}")
    logger.info(f"train ill num:\t {train_ill.shape[0]} \t test ill num:\t {test_ill.shape[0]}")
    logger.info("-------------------------")

    eval_ill = None
    input_idx = torch.LongTensor(np.arange(ENT_NUM))
    train_ill = EADataset(train_ill)
    test_ill = EADataset(test_ill)
    

    return {
        'ent_num': ENT_NUM,
        'rel_num': REL_NUM,
        'images_list': img_features,
        'rel_features': rel_features,
        'att_features': att_features,
        'name_features': name_features,
        'char_features': char_features,
        'input_idx': input_idx,
        'triples': triples,
        'msnea_need': msnea_need
    }, {"left": left_non_train, "right": right_non_train}, train_ill, test_ill, eval_ill, test_ill_

def load_word2vec(path, dim=300):
    """
    glove or fasttext embedding
    """
    # print('\n', path)
    word2vec = dict()
    err_num = 0
    err_list = []

    with open(path, 'r', encoding='utf-8') as file:
        for line in tqdm(file.readlines(), desc="load word embedding"):
            line = line.strip('\n').split(' ')
            if len(line) != dim + 1:
                continue
            try:
                v = np.array(list(map(float, line[1:])), dtype=np.float64)
                word2vec[line[0].lower()] = v
            except:
                err_num += 1
                err_list.append(line[0])
                continue
    file.close()
    print("err list ", err_list)
    print("err num ", err_num)
    return word2vec

def load_char_bigram(path):
    """
    character bigrams of translated entity names
    """
    # load the translated entity names
    ent_names = json.load(open(path, "r"))
    # generate the bigram dictionary
    char2id = {}
    count = 0
    for _, name in ent_names:
        for word in name:
            word = word.lower()
            for idx in range(len(word) - 1):
                if word[idx:idx + 2] not in char2id:
                    char2id[word[idx:idx + 2]] = count
                    count += 1
    return ent_names, char2id

def load_word_char_features(node_size, word2vec_path, args, logger):
    """
    node_size : ent num
    """
    name_path = os.path.join(args.data_path, "DBP15K", "translated_ent_name", "dbp_" + args.data_split + ".json")
    assert osp.exists(name_path)
    save_path_name = os.path.join(args.data_path, "embedding", f"dbp_{args.data_split}_name.pkl")
    save_path_char = os.path.join(args.data_path, "embedding", f"dbp_{args.data_split}_char.pkl")
    if osp.exists(save_path_name) and osp.exists(save_path_char):
        logger.info(f"load entity name emb from {save_path_name} ... ")
        ent_vec = pickle.load(open(save_path_name, "rb"))
        logger.info(f"load entity char emb from {save_path_char} ... ")
        char_vec = pickle.load(open(save_path_char, "rb"))
        return ent_vec, char_vec

    word_vecs = load_word2vec(word2vec_path)
    ent_names, char2id = load_char_bigram(name_path)

    # generate the word-level features and char-level features

    ent_vec = np.zeros((node_size, 300))
    char_vec = np.zeros((node_size, len(char2id)))
    for i, name in ent_names:
        k = 0
        for word in name:
            word = word.lower()
            if word in word_vecs:
                ent_vec[i] += word_vecs[word]
                k += 1
            for idx in range(len(word) - 1):
                
                char_vec[i, char2id[word[idx:idx + 2]]] += 1
        if k:
            ent_vec[i] /= k
        else:
            ent_vec[i] = np.random.random(300) - 0.5

        if np.sum(char_vec[i]) == 0:
            char_vec[i] = np.random.random(len(char2id)) - 0.5
        ent_vec[i] = ent_vec[i] / np.linalg.norm(ent_vec[i])
        char_vec[i] = char_vec[i] / np.linalg.norm(char_vec[i])

    with open(save_path_name, 'wb') as f:
        pickle.dump(ent_vec, f)
    with open(save_path_char, 'wb') as f:
        pickle.dump(char_vec, f)
    logger.info("save entity emb done. ")
    return ent_vec, char_vec

def visual_pivot_induction(args, left_ents, right_ents, img_features, ills, logger):

    l_img_f = img_features[left_ents]  # left images
    r_img_f = img_features[right_ents]  # right images

    img_sim = l_img_f.mm(r_img_f.t())
    topk = args.unsup_k
    two_d_indices = get_topk_indices(img_sim, topk * 100)
    del l_img_f, r_img_f, img_sim

    visual_links = []
    used_inds = []
    count = 0
    for ind in two_d_indices:
        
        
        if left_ents[ind[0]] in used_inds:
            continue
        if right_ents[ind[1]] in used_inds:
            continue
        used_inds.append(left_ents[ind[0]])
        used_inds.append(right_ents[ind[1]])
        
        visual_links.append((left_ents[ind[0]], right_ents[ind[1]]))
        count += 1
        if count == topk:
            break

    count = 0.0
    for link in visual_links:
        if link in ills:
            count = count + 1
    logger.info(f"{(count / len(visual_links) * 100):.2f}% in true links")
    logger.info(f"visual links length: {(len(visual_links))}")
    train_ill = np.array(visual_links, dtype=np.int32)
    return train_ill

# ---------- EVA ----------

def read_raw_data(file_dir, lang=[1, 2]):
    """
    Read DBP15k/DWY15k dataset.
    Parameters
    ----------
    file_dir: root of the dataset.
    Returns
    -------
    ent2id_dict : A dict mapping from entity name to ids
    ills: inter-lingual links (specified by ids)
    triples: a list of tuples (ent_id_1, relation_id, ent_id_2)
    r_hs: a dictionary containing mappings of relations to a list of entities that are head entities of the relation
    r_ts: a dictionary containing mappings of relations to a list of entities that are tail entities of the relation
    ids: all ids as a list
    """
    print('loading raw data...')

    def read_file(file_paths):
        tups = []
        for file_path in file_paths:
            with open(file_path, "r", encoding="utf-8") as fr:
                for line in fr:
                    params = line.strip("\n").split("\t")
                    tups.append(tuple([int(x) for x in params]))
        return tups

    def read_dict(file_paths):
        ent2id_dict = {}
        ids = []
        for file_path in file_paths:
            id = set()
            with open(file_path, "r", encoding="utf-8") as fr:
                for line in fr:
                    params = line.strip("\n").split("\t")
                    ent2id_dict[params[1]] = int(params[0])
                    id.add(int(params[0]))
            ids.append(id)
        return ent2id_dict, ids
    ent2id_dict, ids = read_dict([file_dir + "/ent_ids_" + str(i) for i in lang])

    ent_ids1_dict, _ = read_dict([file_dir + "/ent_ids_1"])
    ent_ids2_dict, _ = read_dict([file_dir + "/ent_ids_2"])
    kg1_entities_list = list(ent_ids1_dict.values())
    kg2_entities_list = list(ent_ids2_dict.values())
    entities_list = list(ent2id_dict.values())

    id_relation_triples1 = read_file([file_dir + "/triples_1"])
    id_relation_triples2 = read_file([file_dir + "/triples_2"])

    ills = read_file([file_dir + "/ill_ent_ids"])
    train_entities1 = [link[0] for link in ills]
    train_entities2 = [link[1] for link in ills]

    rt_dict1, hr_dict1 = uris_relation_triple_2ids(id_relation_triples1)
    rt_dict2, hr_dict2 = uris_relation_triple_2ids(id_relation_triples2)

    triples = id_relation_triples1 + id_relation_triples2

    # sup_triples1_set, sup_triples2_set = generate_sup_relation_triples(ills, rt_dict1, hr_dict1, rt_dict2, hr_dict2)
    # relation_triples_list1, relation_triples_list2 = add_sup_relation_triples(sup_triples1_set, sup_triples2_set, id_relation_triples1, id_relation_triples2)
    
    # relation_triples_set1 = set(relation_triples_list1)
    # relation_triples_set2 = set(relation_triples_list2)
    # relation_triples_set = relation_triples_set1 | relation_triples_set2

    r_hs, r_ts = {}, {}

    for (h, r, t) in triples:
        if r not in r_hs:
            r_hs[r] = set()
        if r not in r_ts:
            r_ts[r] = set()
        r_hs[r].add(h)
        r_ts[r].add(t)
    
    assert len(r_hs) == len(r_ts)
    return {
        "train_entities1": train_entities1,
        "train_entities2": train_entities2,
        "kg1_entities_list": kg1_entities_list,
        "kg2_entities_list": kg2_entities_list,
        "entities_list": entities_list
    }, ent2id_dict, ills, triples, r_hs, r_ts, ids, rt_dict1, hr_dict1, rt_dict2, hr_dict2, id_relation_triples1, id_relation_triples2

def uris_relation_triple_2ids(uris):
    rt_dict, hr_dict = dict(), dict()
    for h_id, r_id, t_id in uris:
        rt_set = rt_dict.get(h_id, set())
        rt_set.add((r_id, t_id))
        rt_dict[h_id] = rt_set
        hr_set = hr_dict.get(t_id, set())
        hr_set.add((h_id, r_id))
        hr_dict[t_id] = hr_set
    return rt_dict, hr_dict

def generate_sup_relation_triples(sup_links, rt_dict1, hr_dict1, rt_dict2, hr_dict2):
    new_triples1, new_triples2 = set(), set()
    
    for ent1, ent2 in sup_links:
        new_triples1 |= (generate_sup_relation_triples_one_link(ent1, ent2, rt_dict1, hr_dict1))
        new_triples2 |= (generate_sup_relation_triples_one_link(ent2, ent1, rt_dict2, hr_dict2))
    print("supervised relation triples: {}, {}".format(len(new_triples1), len(new_triples2)))
    return new_triples1, new_triples2

def add_sup_relation_triples(sup_triples1, sup_triples2, id_relation_triples1, id_relation_triples2):
    id_relation_triples1_set = set(id_relation_triples1)
    id_relation_triples1_set |= sup_triples1
    id_relation_triples2_set = set(id_relation_triples2)
    id_relation_triples2_set |= sup_triples2
    return list(id_relation_triples1_set), list(id_relation_triples2_set)

def generate_sup_relation_triples_one_link(e1, e2, rt_dict, hr_dict):
    new_triples = set()
    for r, t in rt_dict.get(e1, set()):
        new_triples.add((e2, r, t))
    for h, r in hr_dict.get(e1, set()):
        new_triples.add((h, r, e2))
    return new_triples

def loadfile(fn, num=1):
    print('loading a file...' + fn)
    ret = []
    with open(fn, encoding='utf-8') as f:
        for line in f:
            th = line[:-1].split('\t')
            x = []
            for i in range(num):
                x.append(int(th[i]))
            ret.append(tuple(x))
    return ret

def get_ids(fn):
    ids = []
    with open(fn, encoding='utf-8') as f:
        for line in f:
            th = line[:-1].split('\t')
            ids.append(int(th[0]))
    return ids

def get_ent2id(fns):
    ent2id = {}
    for fn in fns:
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                th = line[:-1].split('\t')
                ent2id[th[1]] = int(th[0])
    return ent2id

# The most frequent attributes are selected to save space
def load_attr(fns, e, ent2id, topA=1000):
    cnt = {}
    for fn in fns:
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                th = line[:-1].split('\t')
                
                if th[0] not in ent2id:
                    continue
                
                for i in range(1, len(th)):
                    if th[i] not in cnt:
                        cnt[th[i]] = 1
                    else:
                        cnt[th[i]] += 1
    fre = [(k, cnt[k]) for k in sorted(cnt, key=cnt.get, reverse=True)]
    attr2id = {}
    # pdb.set_trace()
    topA = min(1000, len(fre))
    for i in range(topA):
        attr2id[fre[i][0]] = i
    attr = np.zeros((e, topA), dtype=np.float32)
    for fn in fns:
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                th = line[:-1].split('\t')
                if th[0] in ent2id:
                    for i in range(1, len(th)):
                        if th[i] in attr2id:
                            attr[ent2id[th[0]]][attr2id[th[i]]] = 1.0
    return attr

def load_relation(e, KG, topR=1000):
    # (39654, 1000)
    rel_mat = np.zeros((e, topR), dtype=np.float32)
    
    rels = np.array(KG)[:, 1]
    
    top_rels = Counter(rels).most_common(topR)
    
    rel_index_dict = {r: i for i, (r, cnt) in enumerate(top_rels)}
    for tri in KG:
        h = tri[0]
        r = tri[1]
        o = tri[2]
        if r in rel_index_dict:
            rel_mat[h][rel_index_dict[r]] += 1.
            rel_mat[o][rel_index_dict[r]] += 1.
    return np.array(rel_mat)

def load_json_embd(path):
    embd_dict = {}
    with open(path) as f:
        for line in f:
            example = json.loads(line.strip())
            vec = np.array([float(e) for e in example['feature'].split()])
            embd_dict[int(example['guid'])] = vec
    return embd_dict


# # /home/chenzhuo/data/eva/pkls/fr_en_GA_id_img_feature_dict.pkl
def load_img(logger, e_num, path, ills):
    
    img_dict = pickle.load(open(path, "rb"))
    # init unknown img vector with mean and std deviation of the known's
    imgs_np = np.array(list(img_dict.values()))
    mean = np.mean(imgs_np, axis=0)
    std = np.std(imgs_np, axis=0)
    # img_embd = np.array([np.zeros_like(img_dict[0]) for i in range(e_num)]) # no image
    # img_embd = np.array([img_dict[i] if i in img_dict else np.zeros_like(img_dict[0]) for i in range(e_num)])

    
    
    
    
    img_embd = np.array([img_dict[i] if i in img_dict else np.random.normal(mean, std, mean.shape[0]) for i in range(e_num)])

    
    ent_wo_img = [i for i in range(e_num) if i not in img_dict]
    ent_w_img = [i for i in range(e_num) if i in img_dict]

    # with open("ent_wo_img.json", "w") as fp:
    #     json.dump(ent_wo_img, fp)

    all_ent = [i[0] for i in ills] + [i[1] for i in ills]
    ent_wo_img_ill = [i for i in all_ent if i in ent_wo_img]
    ent_w_img_ill = [i for i in all_ent if i in ent_w_img]
    # pdb.set_trace()
    logger.info(f"{(100 * len(img_dict) / e_num):.2f}% entities have images")
    logger.info(f"{(100 * len(ent_w_img_ill) / len(all_ent)):.2f}% entities in EA dataset have images")

    return img_embd, ent_wo_img, ent_w_img
