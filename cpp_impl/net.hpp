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
 public:
  NTuple(int isom_num = 8) : isom_num(isom_num){};
  NTuple(const std::string& load_path, int isom_num = 8);
  void init(){
    values.fill(new Board::Reward[FEAT_SIZE]);
  }
  ~NTuple() {
    for (auto& v : values) delete[] v;
  };
  void load(const std::string& load_path) override;
  void save(const std::string& save_path) override;
  float evaluate(const Board& b) const override;
  void update(const Board& b, const float error, const float alpha) override;
  auto get_feats(const Board& b) {
    for (int i = 0; i < FEAT_NUM; i++) {
      for (int isom = 0; isom < isom_num; isom++) {
        uint32_t feat = 0;
        for (const auto& f : feat_idx[i]) {
          feat = feat * NUM_RANGE;
          feat += b.get(f, isom);
        }
        co_yield std::make_pair(i, feat);
      }
    }
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

  int isom_num;
};

template <int FEAT_SIZE, int FEAT_NUM>
NTuple<FEAT_SIZE, FEAT_NUM>::NTuple(const std::string& load_path,
                                      int isom_num)
    : isom_num(isom_num) {
  load(load_path);
};

template <int FEAT_SIZE, int FEAT_NUM>
void NTuple<FEAT_SIZE, FEAT_NUM>::save(const std::string& save_path) {
  std::ofstream ofs(save_path);
  if (!ofs.is_open()) {
    std::cout << "Cannot open file " << save_path << std::endl;
    return;
  }
  for (auto& v : values) {
    for (auto& e : v) ofs << e << " ";
    ofs << std::endl;
  }
  for (auto& v : feat_idx) {
    for (auto& e : v) ofs << e << " ";
    ofs << std::endl;
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
  for (auto& v : values) {
    for (auto& e : v) ifs >> e;
  }
  for (auto& v : feat_idx) {
    for (auto& e : v) ifs >> e;
  }
  ifs.close();
};

// evaluate the board
template <int FEAT_SIZE, int FEAT_NUM>
float NTuple<FEAT_SIZE, FEAT_NUM>::evaluate(const Board& b) const {
  float value = 0;
  for (int i = 0; i < FEAT_NUM; i++) {
    for (int isom = 0; isom < isom_num; isom++) {
      uint32_t feat = 0;
      for (const auto& f : feat_idx[i]) {
        feat = feat * NUM_RANGE;
        feat += b.get(f, isom);
      }
      value += values[i][feat];
    }
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
