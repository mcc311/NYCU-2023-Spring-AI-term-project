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

class Net {
 public:
  Net() = default;
  virtual void load(const std::string& load_path){};
  virtual void save(const std::string& save_path){};
  virtual void update(const Board& b, const float error, const float alpha){};
  virtual float evaluate(const Board& b) const {};
};

template <int FEAT_SIZE, int FEAT_NUM>
class NTuple : public Net {
 private:
  static constexpr int NUM_RANGE = 100;
  static constexpr size_t net_size =
      static_cast<size_t>(std::pow(NUM_RANGE, FEAT_SIZE));

 public:
  NTuple(){init();};
  NTuple(const std::string& load_path);
  void init() { values.fill(new Board::Reward[net_size]); }
  ~NTuple() {
    // for (auto& v : values) delete[] v;
  };
  void load(const std::string& load_path) override;
  void save(const std::string& save_path) override;
  float evaluate(const Board& b) const override;
  void update(const Board& b, const float error, const float alpha) override;
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

 private:
  // n tuple value array
  std::array<Board::Reward*, FEAT_NUM> values;
  // feature index array
  std::array<std::array<int, FEAT_SIZE>, FEAT_NUM> feat_idx;

  int isom_num = 8;
};

template <int FEAT_SIZE, int FEAT_NUM>
NTuple<FEAT_SIZE, FEAT_NUM>::NTuple(const std::string& load_path) {
  load(load_path);
};

template <int FEAT_SIZE, int FEAT_NUM>
void NTuple<FEAT_SIZE, FEAT_NUM>::save(const std::string& save_path) {
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
void NTuple<FEAT_SIZE, FEAT_NUM>::load(const std::string& load_path) {
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
Board::Reward NTuple<FEAT_SIZE, FEAT_NUM>::evaluate(const Board& b) const {
  Board::Reward value = 0;
  for (const auto& [i, feat] : get_feats(b)) {
    value += values[i][feat];
  }
  return value / (FEAT_NUM * isom_num);
};

// update the n tuple
template <int FEAT_SIZE, int FEAT_NUM>
void NTuple<FEAT_SIZE, FEAT_NUM>::update(const Board& b, const float error,
                                         const float alpha) {
  for (const auto& [i, feat] : get_feats(b)) {
    values[i][feat] += alpha * error;
  }
};
