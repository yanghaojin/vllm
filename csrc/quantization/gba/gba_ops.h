#pragma once

#include <torch/extension.h>
#include <vector>
#include <tuple>

//GBA quantitative operation statement
torch::Tensor gba_linear_forward(
    torch::Tensor x,
    torch::Tensor qweight,
    torch::Tensor qscales,
    torch::Tensor qzeros,
    torch::Tensor q_perm,
    int group_size,
    int bits,
    bool use_mbw = false,
    torch::Tensor q_group_map = torch::Tensor(),
    std::vector<int> rows = std::vector<int>()
);

std::pair<torch::Tensor, std::vector<int>> gba_trans_qweight(
    torch::Tensor qweight,
    torch::Tensor q_groups,
    bool use_mbw,
    int height,
    int groups,
    int bits
);

torch::Tensor gba_dequantize_weight(
    torch::Tensor qweight,
    torch::Tensor qscales,
    torch::Tensor qzeros,
    torch::Tensor q_perm,
    int group_size,
    int bits,
    bool use_mbw = false,
    torch::Tensor q_group_map = torch::Tensor(),
    std::vector<int> rows = std::vector<int>()
);

// Group mapping tool function
torch::Tensor make_group_map(
    torch::Tensor q_groups,
    int num_qrows
);