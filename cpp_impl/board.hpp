#pragma once
#include <iostream>
#include <vector>

class Board {
 public:
  using Reward = float;
  using Action = int;
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

  const std::tuple<const int (&)[3], const int> action2idx(int action) const {
    return {idxs[action % 6], /*minus= */ action / 6 + 1};
  }
  bool legal(int action) const { //TODO: Use `pext` to accelerate 
    auto&& [idxs, minus] = action2idx(action);
    bool legal = true;
    for (auto& idx : idxs) {
      legal &= get(idx) >= minus;
    }
    return legal;
  };

  std::tuple<Reward, bool> terminated() {
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
    static constexpr Reward bonus = 15;
    static constexpr Reward penalty = -7;

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

  std::tuple<Reward, bool> apply(int action) {
    if(!legal(action)) std::cout << "ILLEGAL!\n";
    auto&& [idxs, minus] = action2idx(action);
    for (auto& idx : idxs) {
      set(idx, get(idx) - minus);
    }
    auto&& [rt, done] = terminated();
    return {rt - minus, done};
  };
  Board& operator=(const Board& b) {
    raw = b.raw;
    return *this;
  };
  int max_min_unit(){
    int max = 0;
    for(const auto& idx : idxs){
      int min = 100;
      for(const auto& i : idx){
        min = std::min(get(i), min);
      }
      max = std::max(max, min);
    }
    return max;
  }
  int min_max_unit(){
    int min = 100;
    for(int i = 0; i < 9; i++) min = std::min(get(i), min);
    return min;
  };

  const std::vector<Action> shuffle_legal_move(int low = 0, int N = 18) const { // TODO: Don't use! this is bugged.
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::vector<int> numbers(N-low);
    std::iota(numbers.begin(), numbers.end(), low);

    // Shuffle the numbers randomly
    std::ranges::shuffle(numbers, gen);
    auto view = numbers | std::views::filter([this](int action) { return legal(action); });
    return {view.begin(), view.end()};
  };

  const std::array<std::tuple<int, int>, 6> min_of_each() const{
    std::array<std::tuple<int, int>, 6> result = {};
    for(int i = 0; i < 6; i++){
      int min = std::numeric_limits<int>::infinity();
      int min_id = 0;
      for(int j = 0; j < 3; j++){
        int val = get(idxs[i][j]);
        if(val < min){
          min = val;
          min_id = j;
        }
      }
      result[i] = {min_id, min};
    }
    return result;
  }

};


std::ostream& operator<<(std::ostream& os, const Board& b) {
  for (int i = 0; i < 9; i++) {
    os << int(b.get(i)) << ((i + 1) % 3 ? "\t" : "\n");
  }
  return os;
}