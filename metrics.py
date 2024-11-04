from collections import Counter
import math
import numpy as np
import torch


def DILAD(rec, label, item_feat, alpha=0.5, similarity="dot", reduction=True):
    """
    @description: novel metric based on ILAD, give different weights to matched and unmatched items
    @param {-} rec, recommendations
    @param {-} label, ground-truth items in test set
    @param {-} item_feat, item embeddings
    @param {-} alpha, parameter of alpha_ILAD
    """
    r = getLabel(label, rec)
    ilad = []
    for cur_rec, cur_r in zip(rec, r):
        # cur_r = cur_r[cur_rec>0]
        # cur_rec = cur_rec[cur_rec>0]
        if len(cur_rec) < 2:
            continue
        ild = 0
        for i in range(len(cur_rec)):
            cur_ilds = []
            for j in range(len(cur_rec)):
                if i != j:
                    w_i = 1 if cur_r[i] else alpha
                    w_j = 1 if cur_r[j] else alpha
                    cur_ilds.append(
                        w_i*w_j * get_similarity(cur_rec[i], cur_rec[j], item_feat, similarity)
                    )
            ild += np.mean(cur_ilds)
        ilad.append(ild / len(cur_rec))

    if reduction:
        return np.sum(ilad)
    else:
        return np.array(ilad)


def extend_category(categories, item_category):
    if type(item_category) == list:
        categories += item_category
    else:
        categories.append(item_category)
    return categories


def DCC(rec, label, item_categories, num_category, alpha=0.5, reduction=True):
    """
    @description: novel metric based on CC,
    @param {-} rec, recommendations
    @param {-} label, ground-truth items in test set
    @param {-} item_categories, item categories
    """
    r = getLabel(label, rec)
    hcc = []
    for cur_rec, cur_r, cur_label in zip(rec, r, label):
        if len(cur_rec) == 0 or len(cur_label) == 0:
            hcc.append(0)
            continue
        pos_categories, neg_categories, label_categories = [], [], []
        for i, r in zip(cur_rec, cur_r):
            if r:
                pos_categories = extend_category(pos_categories, item_categories[i])
            else:
                neg_categories = extend_category(neg_categories, item_categories[i])
        # for i in cur_label:
        #     label_categories = extend_category(label_categories, item_categories[i])
        cur_hcc =len(np.unique(pos_categories)) + alpha*len(
            np.unique(neg_categories)
        )
        # cur_hcc/=len(np.unique(label_categories))
        hcc.append(cur_hcc)
    if reduction:
        return np.sum(hcc)/num_category
    else:
        return np.array(hcc)/num_category


def FDCC(rec, label, item_categories, num_category, alpha=0.5, b=2, reduction=True):
    """
    @description: novel metric based on CC,
    @param {-} rec, recommendations
    @param {-} label, ground-truth items in test set
    @param {-} item_categories, item categories
    """
    r = getLabel(label, rec)
    hfcc = []
    for cur_rec, cur_r, cur_label in zip(rec, r, label):
        if len(cur_rec) == 0 or len(cur_label) == 0:
            hfcc.append(0)
            continue
        pos_categories, neg_categories, label_categories = [], [], []
        for i, r in zip(cur_rec, cur_r):
            if r:
                pos_categories = extend_category(pos_categories, item_categories[i])
            else:
                neg_categories = extend_category(neg_categories, item_categories[i])
        for i in cur_label:
            label_categories = extend_category(label_categories, item_categories[i])
        # pos_counter = Counter(pos_categories)
        # neg_counter = Counter(neg_categories)
        label_counter = Counter(label_categories)
        cur_hfcc = 0
        for cat in np.unique(pos_categories):
            if label_counter[cat]>b:
                # if label_counter[cat] > 10:
                #     print(label_counter[cat], len(label_categories))
                cur_hfcc += math.log(label_counter[cat], b)
            else:
                cur_hfcc += 1
        cur_hfcc += alpha*len(np.unique(neg_categories))
        hfcc.append(cur_hfcc)

    if reduction:
        return np.sum(hfcc)/num_category
    else:
        return np.array(hfcc)/num_category


# ====================Metrics==============================
# https://github.com/gusye1234/LightGCN-PyTorch/blob/master/code/utils.py
def getLabel(test_data, pred_data):
    """
    @description: pred: same size as predictTopK, e.g., [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    """
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype("float")

def get_similarity(i, j, item_feat, similarity):
    if similarity == "jaccard":
        sim = jaccard_similarity(item_feat[i], item_feat[j])
    elif similarity == "dot":
        sim = 1 - np.dot(item_feat[i], item_feat[j])
    return sim


def jaccard_similarity(i_users, j_users):
    i_users = set(i_users)
    j_users = set(j_users)
    intersection = len(i_users.intersection(j_users))
    union = len(i_users.union(j_users))
    return intersection / union if union != 0 else 0
