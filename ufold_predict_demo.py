from collections import defaultdict
import os
import time
import subprocess
from itertools import product

import torch
import numpy as np

from Network import U_Net as FCNNet
from postprocess import postprocess_new_nc, postprocess_new


def get_seq(contact):
    seq = None
    seq = torch.mul(contact.argmax(axis=1), contact.sum(axis=1).clamp_max(1))
    seq[contact.sum(axis=1) == 0] = -1
    return seq


def seq2dot(seq):
    idx = np.arange(1, len(seq) + 1)
    dot_file = np.array(["_"] * len(seq))
    dot_file[seq > idx] = "("
    dot_file[seq < idx] = ")"
    dot_file[seq == 0] = "."
    dot_file = "".join(dot_file)
    return dot_file


def get_ct_dict(predict_matrix, batch_num, ct_dict):

    for i in range(0, predict_matrix.shape[1]):
        for j in range(0, predict_matrix.shape[1]):
            if predict_matrix[:, i, j] == 1:
                if batch_num in ct_dict.keys():
                    ct_dict[batch_num] = ct_dict[batch_num] + [(i, j)]
                else:
                    ct_dict[batch_num] = [(i, j)]
    return ct_dict


def get_ct_dict_fast_one(predict_matrix, seq_embedding, seq_name_save, ct_file_name):
    seq_tmp = (
        torch.mul(
            predict_matrix.cpu().argmax(axis=1),
            predict_matrix.cpu().sum(axis=1).clamp_max(1),
        )
        .numpy()
        .astype(int)
    )
    seq_tmp[predict_matrix.cpu().sum(axis=1) == 0] = -1
    dot_list = seq2dot((seq_tmp + 1).squeeze())
    letter = "AUCG"

    seq_embedding_nonzero = seq_embedding[~torch.all(seq_embedding == 0, axis=1)]
    letter_idx = torch.argmax(seq_embedding_nonzero, dim=1)
    seq_letter = "".join([letter[item] for item in letter_idx])
    seq = ((seq_tmp + 1).squeeze(), torch.arange(predict_matrix.shape[-1]).numpy() + 1)
    ct_dict = [(seq[0][i], seq[1][i]) for i in np.arange(len(seq[0])) if seq[0][i] != 0]
    dot_dict = [seq_letter, dot_list[: len(seq_letter)]]

    ct_file_output(ct_dict, seq_letter, seq_name_save, ct_file_name)

    _, _, noncanonical_pairs = type_pairs(ct_dict, seq_letter)
    tertiary_bp = [list(x) for x in set(tuple(x) for x in noncanonical_pairs)]
    str_tertiary = []
    for i, I in enumerate(tertiary_bp):
        if i == 0:
            str_tertiary += "(" + str(I[0]) + "," + str(I[1]) + '):color=""#FFFF00""'
        else:
            str_tertiary += ";(" + str(I[0]) + "," + str(I[1]) + '):color=""#FFFF00""'

    tertiary_bp = "".join(str_tertiary)
    return dot_dict, tertiary_bp


def ct_file_output(pairs, seq, seq_name_save, ct_file_name):
    col1 = np.arange(1, len(seq) + 1, 1)
    col2 = np.array([i for i in seq])
    col3 = np.arange(0, len(seq), 1)
    col4 = np.append(np.delete(col1, 0), [0])
    col5 = np.zeros(len(seq), dtype=int)

    for i, I in enumerate(pairs):
        col5[I[0] - 1] = int(I[1])

    col6 = np.arange(1, len(seq) + 1, 1)
    temp = np.vstack(
        (
            np.char.mod("%d", col1),
            col2,
            np.char.mod("%d", col3),
            np.char.mod("%d", col4),
            np.char.mod("%d", col5),
            np.char.mod("%d", col6),
        )
    ).T

    np.savetxt(
        ct_file_name,
        (temp),
        delimiter="\t",
        fmt="%s",
        header=">seq length: " + str(len(seq)) + "\t seq name: " + seq_name_save,
        comments="",
    )

    return


def type_pairs(pairs, sequence):
    sequence = [i.upper() for i in sequence]
    AU_pair = []
    GC_pair = []
    GU_pair = []
    other_pairs = []
    for i in pairs:
        if [sequence[i[0] - 1], sequence[i[1] - 1]] in [["A", "U"], ["U", "A"]]:
            AU_pair.append(i)
        elif [sequence[i[0] - 1], sequence[i[1] - 1]] in [["G", "C"], ["C", "G"]]:
            GC_pair.append(i)
        elif [sequence[i[0] - 1], sequence[i[1] - 1]] in [["G", "U"], ["U", "G"]]:
            GU_pair.append(i)
        else:
            other_pairs.append(i)
    watson_pairs_t = AU_pair + GC_pair
    wobble_pairs_t = GU_pair
    other_pairs_t = other_pairs

    return watson_pairs_t, wobble_pairs_t, other_pairs_t


def get_cut_len(data_len, set_len):
    l = data_len
    if l <= set_len:
        l = set_len
    else:
        l = (((l - 1) // 16) + 1) * 16
    return l


def to_AUCG(d):
    try:
        return "AUCG"[list(d).index(1)]
    except Exception:
        return "N"


def creatmat(data, device=None):
    if device == None:
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
    # device = torch.device('cpu')

    with torch.no_grad():
        data = "".join([to_AUCG(d) for d in data])
        paired = defaultdict(
            int, {"AU": 2, "UA": 2, "GC": 3, "CG": 3, "UG": 0.8, "GU": 0.8}
        )

        mat = torch.tensor(
            [[paired[x + y] for y in data] for x in data], dtype=torch.float32
        ).to(device)
        n = len(data)

        i, j = torch.meshgrid(
            torch.arange(n).to(device), torch.arange(n).to(device), indexing="ij"
        )
        t = torch.arange(30).to(device)
        m1 = torch.where(
            (i[:, :, None] - t >= 0) & (j[:, :, None] + t < n),
            mat[
                torch.clamp(i[:, :, None] - t, 0, n - 1),
                torch.clamp(j[:, :, None] + t, 0, n - 1),
            ],
            0,
        )
        m1 *= torch.exp(-0.5 * t * t)

        m1_0pad = torch.nn.functional.pad(m1, (0, 1))
        first0 = torch.argmax((m1_0pad == 0).to(int), dim=2)
        to0indices = t[None, None, :] > first0[:, :, None]
        m1[to0indices] = 0
        m1 = m1.sum(dim=2)

        t = torch.arange(1, 30).to(device)
        m2 = torch.where(
            (i[:, :, None] + t < n) & (j[:, :, None] - t >= 0),
            mat[
                torch.clamp(i[:, :, None] + t, 0, n - 1),
                torch.clamp(j[:, :, None] - t, 0, n - 1),
            ],
            0,
        )
        m2 *= torch.exp(-0.5 * t * t)

        m2_0pad = torch.nn.functional.pad(m2, (0, 1))
        first0 = torch.argmax((m2_0pad == 0).to(int), dim=2)
        to0indices = torch.arange(29).to(device)[None, None, :] > first0[:, :, None]
        m2[to0indices] = 0
        m2 = m2.sum(dim=2)
        m2[m1 == 0] = 0
        return m1 + m2


def preprocess_one(data_seq, data_len):
    perm = list(product(torch.arange(4), torch.arange(4)))
    l = get_cut_len(data_len, 80)
    data_fcn = torch.zeros((16, l, l))

    if l >= 500:
        seq_adj = torch.zeros((l, 4))
        seq_adj[:data_len] = data_seq[:data_len]
        data_seq = seq_adj.int()

    for n, (i, j) in enumerate(perm):
        data_fcn[n, :data_len, :data_len] = torch.matmul(
            data_seq[:data_len, i].unsqueeze(1), data_seq[:data_len, j].unsqueeze(0)
        )

    data_fcn_1 = torch.zeros((1, l, l))
    data_fcn_1[0, :data_len, :data_len] = creatmat(data_seq[:data_len])

    data_fcn_2 = torch.cat((data_fcn, data_fcn_1), dim=0)

    seq_embeddings = data_fcn_2
    seq_ori = data_seq[:l]

    return seq_embeddings, seq_ori


def one_hot_600(seq_item):
    RNN_seq = seq_item
    BASES = "AUCG"
    bases = np.array([base for base in BASES])
    feat = np.concatenate(
        [
            (
                [(bases == base.upper()).astype(int)]
                if str(base).upper() in BASES
                else np.array([[-1] * len(BASES)])
            )
            for base in RNN_seq
        ]
    )
    if len(seq_item) <= 600:
        one_hot_matrix_600 = np.zeros((600, 4)).astype(int)
    else:
        one_hot_matrix_600 = np.zeros((len(seq_item), 4)).astype(int)
    one_hot_matrix_600[: len(seq_item),] = feat

    return one_hot_matrix_600


def test_one(contact_net, file_name, save_dir, seq_name_save=None, nc=False):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    contact_net.train()

    if nc:
        postprocess = postprocess_new_nc
    else:
        postprocess = postprocess_new

    input_data = open(file_name, "r").readlines()
    assert len(input_data) == 2
    assert input_data[0].startswith(">")
    seq_name = input_data[0][1:].strip()
    seq_data = input_data[1].strip().upper().replace("T", "U")
    assert seq_data.startswith(("A", "U", "C", "G"))

    if seq_name_save is None:
        seq_name_save = seq_name.replace("/", "_")

    seq_length = len(seq_data)
    data_one_hot = torch.Tensor(one_hot_600(seq_data)).int()

    seq_embeddings, seq_ori = preprocess_one(data_one_hot, seq_length)

    seq_embeddings = torch.Tensor(seq_embeddings).unsqueeze(0)
    seq_ori = torch.Tensor(seq_ori).unsqueeze(0)

    seq_embedding_input = torch.Tensor(seq_embeddings.float()).to(device)
    seq_ori = torch.Tensor(seq_ori.float()).to(device)

    with torch.no_grad():
        pred_contacts = contact_net(seq_embedding_input)

    # only post-processing without learning
    u_no_train = postprocess(pred_contacts, seq_ori, 0.01, 0.1, 100, 1.6, True, 1.5)
    map_no_train = (u_no_train > 0.5).float()

    ct_file_name = os.path.join(save_dir, "{}.ct".format(seq_name_save))
    dot_dict, tertiary_bp = get_ct_dict_fast_one(
        map_no_train, seq_ori.squeeze(0), seq_name_save, ct_file_name
    )

    if not nc:
        png_file_name = os.path.join(save_dir, "{}_radiate.png".format(seq_name_save))
        subprocess.Popen(
            [
                "java",
                "-cp",
                "VARNAv3-93.jar",
                "fr.orsay.lri.varna.applications.VARNAcmd",
                "-i",
                ct_file_name,
                "-o",
                png_file_name,
                "-algorithm",
                "radiate",
                "-resolution",
                "8.0",
                "-bpStyle",
                "lw",
            ],
            stderr=subprocess.STDOUT,
            stdout=subprocess.PIPE,
        ).communicate()[0]
    else:
        png_file_name = os.path.join(
            save_dir, "{}_radiatenew.png".format(seq_name_save)
        )
        subprocess.Popen(
            [
                "java",
                "-cp",
                "VARNAv3-93.jar",
                "fr.orsay.lri.varna.applications.VARNAcmd",
                "-i",
                ct_file_name,
                "-o",
                png_file_name,
                "-algorithm",
                "radiate",
                "-resolution",
                "8.0",
                "-bpStyle",
                "lw",
                "-auxBPs",
                tertiary_bp,
            ],
            stderr=subprocess.STDOUT,
            stdout=subprocess.PIPE,
        ).communicate()[0]

    dot_file_name = os.path.join(save_dir, "{}_dot.txt".format(seq_name_save))
    with open(dot_file_name, "w") as f:
        f.write(">{}\n".format(seq_name_save))
        f.write("{}\n".format(dot_dict[0]))
        f.write("{}\n".format(dot_dict[1]))
        f.write("\n")


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    contact_net = FCNNet(img_ch=17)
    MODEL_SAVED = "models/ufold_train_alldata.pt"
    print("==========Start Loading Pretrained Model==========")
    contact_net.load_state_dict(torch.load(MODEL_SAVED, map_location="cuda:0"))
    contact_net.to(device)
    print("==========Finish Loading Pretrained Model==========")
    file_name = "data/input.txt"
    out_dir = "out"
    start_t = time.time()
    test_one(contact_net, file_name, out_dir, "demo")
    end_t = time.time()
    print("cost time {}".format(end_t - start_t))

    print(
        "==========Done!!! Please check results folder for the predictions!=========="
    )
