import torch

def Eu_dis(x):
    device = x.device  # 获取输入张量所在的设备
    """
    Calculate the distance among each row of x
    :param x: N X D
                N: the object number
                D: Dimension of the feature
    :return: N X N distance matrix
    """
    aa = torch.sum(x * x, dim=1)
    ab = torch.matmul(x, x.t())
    dist_mat = aa.unsqueeze(1) + aa - 2 * ab
    dist_mat[dist_mat < 0] = 0
    dist_mat = torch.sqrt(dist_mat)
    dist_mat = torch.max(dist_mat, dist_mat.t())
    return dist_mat


def feature_concat(*F_list, normal_col=False):
    device = F_list[0].device  # 获取输入张量所在的设备
    """
    Concatenate multiple modality feature. If the dimension of a feature matrix is more than two,
    the function will reduce it into two dimension(using the last dimension as the feature dimension,
    the other dimension will be fused as the object dimension)
    :param F_list: Feature matrix list
    :param normal_col: normalize each column of the feature
    :return: Fused feature matrix
    """
    features = torch.empty((0,), device=device)
    for f in F_list:
        if f is not None and f != []:
            # deal with the dimension that more than two
            if f.dim() > 2:
                f = f.reshape(-1, f.shape[-1])
            # normal each column
            if normal_col:
                f_max = torch.max(torch.abs(f), dim=0).values
                f = f / f_max
            # facing the first feature matrix appended to fused feature matrix
            if features.numel() == 0:
                features = f
            else:
                features = torch.cat((features, f), dim=1)
    if normal_col:
        features_max = torch.max(torch.abs(features), dim=0).values
        features = features / features_max
    return features


def hyperedge_concat(*H_list):
    device = H_list[0].device  # 获取输入张量所在的设备
    """
    Concatenate hyperedge group in H_list
    :param H_list: Hyperedge groups which contain two or more hypergraph incidence matrix
    :return: Fused hypergraph incidence matrix
    """
    # H = None
    H = torch.empty((0,), device=device)
    for h in H_list:
        if h is not None and h != []:
            # for the first H appended to fused hypergraph incidence matrix
            if H.numel() == 0:
                H = h
            else:
                if isinstance(h, list):
                    tmp = torch.empty((0,), device=device)
                    for a, b in zip(H, h):
                        tmp.append(torch.cat((a, b), dim=1))
                    H = tmp
                else:
                    H = torch.cat((H, h), dim=1)
    return H


def generate_G_from_H(H, variable_weight=False):
    device = H.device  # 获取输入张量所在的设备
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    if isinstance(H, list):
        G = torch.empty((0,), device=device)
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G
    else:
        return _generate_G_from_H(H, variable_weight)


def _generate_G_from_H(H, variable_weight=False):
    device = H.device  # 获取输入张量所在的设备
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    H = H.to(device)
    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = torch.ones(n_edge, device=device)
    # the degree of the node
    DV = torch.sum(H * W, dim=1)
    # the degree of the hyperedge
    DE = torch.sum(H, dim=0)

    invDE = torch.diag(torch.pow(DE, -1))
    DV2 = torch.diag(torch.pow(DV, -0.5))
    W = torch.diag(W)
    H = H.to(device)
    HT = H.t()

    if variable_weight:
        DV2_H = DV2 @ H
        invDE_HT_DV2 = invDE @ HT @ DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 @ H @ W @ invDE @ HT @ DV2
        return G


def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=True, m_prob=1):
    device = dis_mat.device  # 获取输入张量所在的设备
    """
    construct hypergraph incidence matrix from hypergraph node distance matrix
    :param dis_mat: node distance matrix
    :param k_neig: K nearest neighbor
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object X N_hyperedge
    """
    n_obj = dis_mat.shape[0]
    # construct hyperedge from the central feature space of each node
    n_edge = n_obj
    H = torch.zeros((n_obj, n_edge), device=device)
    for center_idx in range(n_obj):
        dis_mat[center_idx, center_idx] = 0
        dis_vec = dis_mat[center_idx]
        nearest_idx = torch.argsort(dis_vec).squeeze()
        avg_dis = torch.mean(dis_vec)
        if not torch.any(nearest_idx[:k_neig] == center_idx):
            nearest_idx[k_neig - 1] = center_idx

        for node_idx in nearest_idx[:k_neig]:
            if is_probH:
                # H[node_idx, center_idx] = torch.exp(-dis_vec[node_idx] ** 2 / (m_prob * avg_dis) ** 2)
                H[node_idx, center_idx] = torch.exp((-dis_vec[node_idx] ** 2) / ((m_prob * avg_dis) ** 2))

            else:
                H[node_idx, center_idx] = 1.0

    return H


def construct_H_with_KNN(X, K_neigs=[10], split_diff_scale=False, is_probH=True, m_prob=1):
    device = X.device  # 获取输入张量所在的设备
    """
    init multi-scale hypergraph Vertex-Edge matrix from original node feature matrix
    :param X: N_object x feature_number
    :param K_neigs: the number of neighbor expansion
    :param split_diff_scale: whether split hyperedge group at different neighbor scale
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object x N_hyperedge
    """
    if X.dim() != 2:
        X = X.reshape(-1, X.shape[-1])

    if isinstance(K_neigs, int):
        K_neigs = [K_neigs]

    dis_mat = Eu_dis(X)
    H = torch.empty((0,), device=device)
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
        if not split_diff_scale:
            H = hyperedge_concat(H, H_tmp)
        else:
            H.append(H_tmp)
    return H
