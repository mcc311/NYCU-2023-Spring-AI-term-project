#pragma once
#include <iostream>
#include <vector>

class board {
 public:
  using reward = int;
  using action = int;
  board() : raw(0){};
  uint64_t raw;
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
  std::tuple<const int (&)[3], const int> action2idx(int action);
  bool legal(int action);
  std::tuple<reward, bool> terminated();
  std::tuple<reward, bool> apply(int action);
  board& operator=(const board& b);
};

std::ostream& operator<<(std::ostream& os, const board& b);