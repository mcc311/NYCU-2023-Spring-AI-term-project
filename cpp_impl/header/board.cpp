#include "board.hpp"


std::tuple<const int (&)[3], const int> board::action2idx(int action) {
  static constexpr int idxs[6][3] = {{0, 1, 2}, {3, 4, 5}, {6, 7, 8},
                                     {0, 3, 6}, {1, 4, 7}, {2, 5, 8}};
  return {idxs[action % 6], /*minus= */ action / 6 + 1};
}
bool board::legal(int action) {
  auto&& [idxs, minus] = action2idx(action);
  bool legal = true;
  for (auto& idx : idxs) {
    legal &= get(idx) >= minus;
  }
  return legal;
};

std::tuple<board::reward, bool> board::terminated() {
  static constexpr uint64_t bonus_pattern[8] = {
      9151318806505701376ULL,  // 1st col
      71494678175825792ULL,    // 2nd col
      558552173248639ULL,      // 3rd col
      9223367638808264704ULL,  // 1st row
      4398044413952ULL,        // 2nd row
      2097151ULL,              // 3rd row
      9151314476908150911ULL,  // diag
      558586000293888ULL,      // flipped-diag
  };
  static constexpr board::reward bonus = 15;
  static constexpr board::reward penalty = -7;

  // check if it's terminated with bonus
  bool is_bonus = false;
  for (auto& pattern : bonus_pattern) {
    is_bonus |= !(raw & pattern);
  }
  // check if it's terminated with penalty
  bool is_penalty = false;
  int count = 0;
  for (int i = 0; i < 6; i++) {
    is_penalty |= legal(i);
  }
  is_penalty = !is_penalty;
  return {bonus * is_bonus + penalty * (is_penalty & !is_bonus),
          is_bonus || is_penalty};
};

std::tuple<board::reward, bool> board::apply(int action) {
  auto&& [idxs, minus] = action2idx(action);
  for (auto& idx : idxs) {
    set(idx, get(idx) - minus);
  }
  auto&& [rt, done] = terminated();
  return {rt - minus, done};
};
board& board::operator=(const board& b) {
  raw = b.raw;
  return *this;
};

std::ostream& operator<<(std::ostream& os, const board& b){
  for (int i = 0; i < 9; i++) {
    os << int(b.get(i)) << ((i + 1) % 3 ? "\t" : "\n");
  }
  return os;
}


