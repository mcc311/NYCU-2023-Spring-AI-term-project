#pragma once
#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <ranges>
#include <vector>

#include "board.hpp"
#include "utils.hpp"

template <int FEAT_SIZE, int FEAT_NUM>
class TupleNet {
 private:
  static constexpr int NUM_RANGE = 100;
  static constexpr size_t net_size =
      static_cast<size_t>(std::pow(NUM_RANGE, FEAT_SIZE));

 public:
  TupleNet() {
    for (auto& v : values) {
      v = new Board::Reward[net_size];
      std::fill(v, v + net_size, 0);
    }
    weights.fill(1.0f);
  };
  TupleNet(const std::string& load_path);
  TupleNet(const std::array<std::array<int, FEAT_SIZE>, FEAT_NUM>& feat_idx_)
      : TupleNet() {
    feat_idx = feat_idx_;
  };
  ~TupleNet(){
      // for (auto& v : values) delete[] v;
  };
  void load(const std::string& load_path);
  void save(const std::string& save_path);
  Board::Reward evaluate(const Board& b) const;
  void update_net(const Board& b, const Board::Reward error,
                  const Board::Reward lr);
  void update_weights(const Board& b, const Board::Reward error,
                      const Board::Reward lr, const Board::Reward lambda);
  auto get_feats(const Board& b) const {
    std::vector<std::pair<int, uint32_t>> result;
    for (int i = 0; i < FEAT_NUM; i++) {
      for (int isom = 0; isom < isom_num; isom++) {
        uint32_t feat = 0;
        for (const auto& f : feat_idx[i]) {
          feat = feat * NUM_RANGE;
          feat += b.get(f, isom);
        }
        result.emplace_back(i, feat);
      }
    }
    return result;
  }
  void set_feats(
      const std::array<std::array<int, FEAT_SIZE>, FEAT_NUM>& feat_idx_) {
    feat_idx = feat_idx_;
  }

 public:
  // n tuple value array
  std::array<Board::Reward*, FEAT_NUM> values;
  // feature index array
  std::array<std::array<int, FEAT_SIZE>, FEAT_NUM> feat_idx;

  std::array<Board::Reward, FEAT_NUM> weights;

  std::array<Board::Reward, FEAT_NUM> gradient_weights;

  int isom_num = 8;
};

template <int FEAT_SIZE, int FEAT_NUM>
TupleNet<FEAT_SIZE, FEAT_NUM>::TupleNet(const std::string& load_path)
    : TupleNet() {
  load(load_path);
};

template <int FEAT_SIZE, int FEAT_NUM>
void TupleNet<FEAT_SIZE, FEAT_NUM>::save(const std::string& save_path) {
  std::ofstream ofs(save_path,
                    std::ios::out | std::ios::binary | std::ios::trunc);
  if (!ofs.is_open()) {
    std::cout << "Cannot open file " << save_path << std::endl;
    return;
  }
  for (auto& v : feat_idx) {
    ofs.write(reinterpret_cast<char*>(v.data()), sizeof(int) * FEAT_SIZE);
  }
  for (auto& v : values) {
    ofs.write(reinterpret_cast<char*>(v), sizeof(Board::Reward) * net_size);
  }

  ofs.close();
};

template <int FEAT_SIZE, int FEAT_NUM>
void TupleNet<FEAT_SIZE, FEAT_NUM>::load(const std::string& load_path) {
  std::ifstream ifs(load_path);
  if (!ifs.is_open()) {
    std::cout << "Cannot open file " << load_path << std::endl;
    return;
  }
  for (auto& v : feat_idx) {
    ifs.read(reinterpret_cast<char*>(v.data()), sizeof(int) * FEAT_SIZE);
  }
  for (auto& v : values) {
    v = new Board::Reward[net_size];
    ifs.read(reinterpret_cast<char*>(v), sizeof(Board::Reward) * net_size);
  }
  ifs.close();
};

// evaluate the board
template <int FEAT_SIZE, int FEAT_NUM>
Board::Reward TupleNet<FEAT_SIZE, FEAT_NUM>::evaluate(const Board& b) const {
  Board::Reward value = 0;
  for (const auto& [i, feat] : get_feats(b)) {
    value += values[i][feat] * weights[i];
  }
  return value;
};

// update the n tuple
template <int FEAT_SIZE, int FEAT_NUM>
void TupleNet<FEAT_SIZE, FEAT_NUM>::update_net(const Board& b,
                                               const Board::Reward error,
                                               const Board::Reward lr) {
  for (const auto& [i, feat] : get_feats(b)) {
    values[i][feat] += lr * error / (FEAT_NUM * isom_num);
  }
};

// update the n tuple
template <int FEAT_SIZE, int FEAT_NUM>
void TupleNet<FEAT_SIZE, FEAT_NUM>::update_weights(const Board& b,
                                                   const Board::Reward error,
                                                   const Board::Reward lr,
                                                   const Board::Reward lambda) {
  gradient_weights.fill(0.0f);
  for (const auto& [i, feat] : get_feats(b)) {
    gradient_weights[i] +=
        2 * (-error) * values[i][feat] + (2 * lambda * weights[i]) / (isom_num);
  }
  for (int i = 0; i < FEAT_NUM; i++) {
    weights[i] -= lr * gradient_weights[i];
  }
};
