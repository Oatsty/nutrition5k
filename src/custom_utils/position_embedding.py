import torch


def get_pos_embed_1d(n_queries: int, emb_dim: int) -> torch.Tensor:
    assert emb_dim % 2 == 0
    pos = torch.arange(0, n_queries, dtype=torch.float).unsqueeze(1)
    denom = torch.exp(
        torch.arange(0, emb_dim, 2, dtype=torch.float)
        / (emb_dim)
        * torch.log(torch.tensor(10000.0))
    )
    pos_embed = torch.empty((n_queries, emb_dim))
    pos_embed[:, 0::2] = torch.sin(pos / denom)
    pos_embed[:, 1::2] = torch.cos(pos / denom)
    return pos_embed


def get_pos_embed_nd(*query_shape: int, emb_dim: int) -> torch.Tensor:
    query_dim = len(query_shape)
    assert emb_dim % query_dim == 0, f"invalid embedding dimension: {emb_dim}"
    hid_dim = emb_dim // query_dim
    pos_emb_list = []
    for i, n_queries in enumerate(query_shape):
        emb = torch.ones(n_queries, emb_dim)
        start = int(hid_dim * i)
        end = int(hid_dim * (i + 1))
        emb[:, start:end] = get_pos_embed_1d(n_queries, hid_dim)
        pos_emb_list.append(emb)
    pos_1 = pos_emb_list.pop(0)
    pos_2 = pos_emb_list.pop(0)
    pos_emb = torch.einsum("...j, ij -> ...ij", pos_1, pos_2)
    while len(pos_emb_list):
        pos_emb = torch.einsum("...j, ij -> ...ij", pos_emb, pos_emb_list.pop(0))
    return pos_emb
