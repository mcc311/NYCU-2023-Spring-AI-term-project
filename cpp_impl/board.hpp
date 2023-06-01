#pragma once
#include <iostream>
#include <vector>

class board {
 public:
  using reward = float;
  using action = int;
  uint64_t raw = 0;
  inline int get(const int index, const int isomorphic = 0) const noexcept {
    static constexpr int isom_table[8][9] = {
        {0, 1, 2, 3, 4, 5, 6, 7, 8}, {2, 5, 8, 1, 4, 7, 0, 3, 6},
        {8, 7, 6, 5, 4, 3, 2, 1, 0}, {6, 3, 0, 7, 4, 1, 8, 5, 2},
        {2, 1, 0, 5, 4, 3, 8, 7, 6}, {0, 3, 6, 1, 4, 7, 2, 5, 8},
        {6, 7, 8, 3, 4, 5, 0, 1, 2}, {8, 5, 2, 7, 4, 1, 6, 3, 0}};
    return int((raw >> (isom_table[isomorphic][index] * 7)) & 0b1111111ull);
  }
  inline void set(const uint8_t index, uint64_t value) {
    uint64_t mask = ~(0b1111111ull << (index * 7));
    raw = (raw & mask) | (value << (index * 7));
  }

  static constexpr int idxs[6][3] = {{0, 1, 2}, {3, 4, 5}, {6, 7, 8},
                                       {0, 3, 6}, {1, 4, 7}, {2, 5, 8}};

  std::tuple<const int (&)[3], const int> action2idx(int action) {
    return {idxs[action % 6], /*minus= */ action / 6 + 1};
  }
  bool legal(int action) { //TODO: Use `pext` to accelerate 
    auto&& [idxs, minus] = action2idx(action);
    bool legal = true;
    for (auto& idx : idxs) {
      legal &= get(idx) >= minus;
    }
    return legal;
  };

  std::tuple<reward, bool> terminated() {
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
    static constexpr reward bonus = 15;
    static constexpr reward penalty = -7;

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
  }

  std::tuple<reward, bool> apply(int action) {
    auto&& [idxs, minus] = action2idx(action);
    for (auto& idx : idxs) {
      set(idx, get(idx) - minus);
    }
    auto&& [rt, done] = terminated();
    return {rt - minus, done};
  };
  board& operator=(const board& b) {
    raw = b.raw;
    return *this;
  };
  int max_min_unit(){
    int max = 0;
    for(const auto& idx : idxs){
      int min = 100;
      for(const auto& i : idx){
        int num = get(i);
        min = (min > num) ? num : min;
      }
      max = (max < min) ? min : max;
    }
    return max;
  }
  int min_max_unit(){
    int min = 100;
    for(int i = 0; i < 9; i++) min =  (min > get(i)) ? get(i) : min;
    return min;
  };

};


std::ostream& operator<<(std::ostream& os, const board& b) {
  for (int i = 0; i < 9; i++) {
    os << int(b.get(i)) << ((i + 1) % 3 ? "\t" : "\n");
  }
  return os;
}