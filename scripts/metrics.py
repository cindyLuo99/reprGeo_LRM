import torch
import torch.nn.functional as F
from scipy.stats import spearmanr # Added import for Pearson correlation
import numpy as np

######## Analysis ########

# 1. Given a set of activations [e.g. torch.Size([30000, 1000])] and their labels
# calculate the prototypes of each category based on the labels
def compute_prototypes(activations, labels, unique_labels):
    prototypes = []
    for label in unique_labels:
        category_activations = activations[labels == label]
        prototype = category_activations.mean(dim=0)  # Compute mean for category prototype
        prototypes.append(prototype)
    return torch.stack(prototypes)  # Shape: [num_categories, feature_dim]

# 2. Compute distances between instances and prototypes before and after modulation
def compute_exemplar_distances(activations, prototypes, labels, unique_labels, distance_metric="cosine"):
    distances = []
    for i, prototype in enumerate(prototypes):
        category_activations = activations[labels == unique_labels[i]]
        # Calculate cosine similarity or distance
        if distance_metric == "cosine": # [0,2]
            dist = 1 - F.cosine_similarity(category_activations, prototype.unsqueeze(0), dim=1)  # Cosine distance

        elif distance_metric == "euclidean": #[0, \infty)
            dist = torch.linalg.vector_norm(category_activations - prototype.unsqueeze(0), dim=1)  # L2 norm  # L2 norm
        distances.append(dist)
    return torch.cat(distances)


# 3. Calculate recall at k (percentage of preserved neighbors) &
#    How much of the shared neighbors are from the same category?
def get_cosine_sim_mtx(activations, device):
    activations = activations.to(device)
    act_normalized = F.normalize(activations, dim=1)
    sim_act = torch.mm(act_normalized, act_normalized.T)
    return sim_act


def get_recall_and_shared_cat_preserve(
    sim1, sim3, labels, device, return_cate=False, ks=range(2,301)
):
    N = sim1.size(0)
    max_k = max(ks)

    # Ensure on GPU
    sim1, sim3, labels = sim1.to(device), sim3.to(device), labels.to(device)

    # Top-k once
    _, all1 = sim1.topk(max_k, dim=1)
    _, all3 = sim3.topk(max_k, dim=1)

    recall = torch.zeros(len(ks), device=device)
    cat_pres = torch.zeros(len(ks), device=device)

    arange = torch.arange(N, device=device).unsqueeze(1).expand(-1, max_k-1)

    for idx, k in enumerate(ks):
        nn1k = all1[:, 1:k]  # [N, k-1] # exclude the vector itself
        nn3k = all3[:, 1:k]

        # preserved neighbor mask
        mask = (nn1k.unsqueeze(2) == nn3k.unsqueeze(1)).any(dim=2)  # [N, k-1]
        tot_pres = mask.sum()  # total preserved across all i

        recall[idx] = 100.0 * tot_pres / ((k-1) * N)

        # category preservation
        if return_cate:
            shared_idx = nn1k[mask]            # (tot_pres,)
            q_idx      = arange[:, :k-1][mask] # (tot_pres,)
            same_cat   = (labels[shared_idx] == labels[q_idx]).sum()
            cat_pres[idx] = 100.0 * same_cat / tot_pres if tot_pres>0 else 0

    return recall.cpu().numpy(), cat_pres.cpu().numpy()

# 4. How many of the neighbors from act_1/act_3 belong to the same category? 
# return (# of same‐category neighbors within k) / k
def get_category_preservation(sim_act, labels, max_k=300):
    device = sim_act.device
    N = sim_act.size(0)
    # 1) get all neighbors once
    _, all_neighbors = sim_act.topk(max_k, dim=1)
    # 2) pre‐alloc
    pres = torch.zeros(max_k+1, device=device)
    labels = labels.to(device)

    for k in range(2, max_k+1):
        nnk     = all_neighbors[:, 1:k]                    # [N, k-1]
        same_ct = (labels.unsqueeze(1) == labels[nnk]).sum()  # scalar tensor
        pres[k] = 100.0 * same_ct / (N * (k-1))

    return pres[2:].cpu().numpy().tolist()



# 5. Given two N×N cosine‐similarity matrices (before & after), compute a C×C tensor M where:
    # M[i,j] = mean_{a∈cat_i, b∈cat_j} [ d_after(a,b) − d_before(a,b) ]
# with d = 1 − sim.

def get_cate_distance_change_matrix(sim_base, sim_mod, labels, unique_labels):
    device = sim_base.device
    labels = labels.to(device)
    
    # 1) distances = 1 - sims
    dist_before = 1.0 - sim_base
    dist_after  = 1.0 - sim_mod
    
    # 2) figure out which categories to include
    if unique_labels is None:
        cats = torch.unique(labels)
    else:
        cats = torch.tensor(unique_labels, device=device)
    C = cats.numel()
    
    # 3) precompute indices for each cat
    idxs = [(labels == c).nonzero(as_tuple=True)[0] for c in cats]
    
    # 4) allocate output
    changes = torch.zeros((C, C), device=device)
    
    # 5) fill in
    for i, ii in enumerate(idxs):
        # ii is a 1D tensor of all image‐indices in category i
        for j, jj in enumerate(idxs):
            # take the block dist[ ii, : ][ :, jj ] via ix_
            block_before = dist_before[ii.unsqueeze(1), jj]  # shape [ni, nj]
            block_after  = dist_after [ii.unsqueeze(1), jj]
            # mean over all pairs
            changes[i, j] = (block_after - block_before).mean()
    
    return changes

# 6. For each category i, compute shift = 1 – similarity(prot_base[i], prot_mod[i]).
#    Returns a (C,) tensor of shift distances.

def compute_prototype_shifts(
    prot_base,
    prot_mod,
    metric='cosine'
):
    """
    For each category i, compute shift = 1 - similarity(prot_base[i], prot_mod[i]).
    Returns a (C,) tensor of shift distances.
    """
    if metric != 'cosine':
        raise NotImplementedError("Only 'cosine' supported")
    sims = F.cosine_similarity(prot_base, prot_mod, dim=1)
    return 1.0 - sims

def compute_between_prototype_changes(
    prot_base,
    prot_mod,
    metric='cosine'
):
    """
    Compute two (M,) vectors, where M = C*(C-1)/2:
      - baseline_seps: all upper-triangle distances from prot_base
      - delta_seps:   prot_mod_distances - prot_base_distances
    """
    if metric != 'cosine':
        raise NotImplementedError("Only 'cosine' supported")
    C = prot_base.size(0)
    # compute full cosine-sim matrices
    sim_b = F.cosine_similarity(prot_base.unsqueeze(1), prot_base.unsqueeze(0), dim=-1)
    sim_m = F.cosine_similarity(prot_mod.unsqueeze(1),  prot_mod.unsqueeze(0),   dim=-1)
    dist_b = 1. - sim_b
    dist_m = 1. - sim_m

    # upper‐triangle indices
    idx = torch.triu_indices(C, C, offset=1, device=prot_base.device)
    vec_b = dist_b[idx[0], idx[1]]
    vec_m = dist_m[idx[0], idx[1]]
    return vec_b, vec_m - vec_b


def calculate_rsa_torch(
    prot_base,
    prot_mod,
    metric='cosine'
):
    """
    Spearman correlation between the upper-triangle RDMs of two prototype sets.
    Returns (rho, p-value).
    """
    if prot_base.shape != prot_mod.shape:
        raise ValueError("Shapes must match")
    C = prot_base.size(0)
    if C < 2:
        return None, None

    if metric != 'cosine':
        raise NotImplementedError("Only 'cosine' supported")

    sim1 = F.cosine_similarity(prot_base.unsqueeze(1), prot_base.unsqueeze(0), dim=-1)
    sim2 = F.cosine_similarity(prot_mod.unsqueeze(1), prot_mod.unsqueeze(0), dim=-1)
    d1 = (1 - sim1).cpu().numpy()
    d2 = (1 - sim2).cpu().numpy()

    # extract upper triangles
    triu_idx = np.triu_indices(C, k=1)
    vec1 = d1[triu_idx]
    vec2 = d2[triu_idx]
    # spearman
    rho, p = spearmanr(vec1, vec2)
    return rho, p


def cross_validation_analysis(
    act_base, act_mod,
    labels, unique_labels,
    k=5, distance_metric="cosine"
):
    """
    For each category, 5-fold CV:
      - compute prototype shifts (global)
      - compute mean change in exemplar→prototype distance (local)
    Returns four dicts keyed by category:
      1. avg local difference
      2. avg global shift
      3. list of local diffs per fold
      4. list of global shifts per fold
    """
    device = act_base.device
    local_diffs = {c: [] for c in unique_labels}
    global_shifts = {c: [] for c in unique_labels}

    for c in unique_labels:
        idx = (labels == c).nonzero(as_tuple=True)[0]
        n   = idx.numel()
        perm= torch.randperm(n, generator=torch.Generator(device=device).manual_seed(42))
        idx = idx[perm]
        fold_size = n // k

        for fold in range(k):
            start = fold*fold_size
            if fold == k-1:
                test_idx  = idx[start:]
            else:
                test_idx  = idx[start:start+fold_size]
            train_idx = idx[~((idx.unsqueeze(1)==test_idx).any(1))]

            pb = act_base[train_idx].mean(0)
            pm = act_mod[train_idx].mean(0)
            # global shift
            if distance_metric=="cosine":
                shift = 1 - F.cosine_similarity(pb.unsqueeze(0), pm.unsqueeze(0)).item()
            else:
                shift = (pb-pm).norm().item()
            global_shifts[c].append(shift)

            # local distances
            tb = act_base[test_idx]
            ta = act_mod[test_idx]
            if distance_metric=="cosine":
                db = 1 - F.cosine_similarity(tb, pb.unsqueeze(0), dim=1)
                da = 1 - F.cosine_similarity(ta, pm.unsqueeze(0), dim=1)
            else:
                db = (tb - pb.unsqueeze(0)).norm(dim=1)
                da = (ta - pm.unsqueeze(0)).norm(dim=1)
            local_diffs[c].append((da - db).mean().item())

    local_avg  = {c: float(np.mean(local_diffs[c]))  for c in unique_labels}
    global_avg = {c: float(np.mean(global_shifts[c])) for c in unique_labels}
    return local_avg, global_avg, local_diffs, global_shifts