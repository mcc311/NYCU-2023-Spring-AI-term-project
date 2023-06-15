#pragma once
#include <iostream>
#include <vector>
class Episode;
class Board {
  friend class Episode;
  friend std::istream& operator>>(std::istream& is, Board& b);
 private:
  uint64_t raw = 0;

 public:
  Board(uint64_t raw = 0) : raw(raw){};
  Board(std::vector<std::vector<int> > board) {
    for (int i = 0; i < 9; i++) {
      set(i, board[i / 3][i % 3]);
    }
  };
  using Reward = float;
  using Action = int;
  using Hash = uint64_t;

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

  inline const std::tuple<const int (&)[3], const int> action2idx(
      int action) const {
    return {idxs[action % 6], /*minus= */ action / 6 + 1};
  }
  inline const bool legal(int action) const {  // TODO: Use `pext` to accelerate
    auto&& [idxs, minus] = action2idx(action);
    bool legal = true;
    for (auto& idx : idxs) {
      legal &= get(idx) >= minus;
    }
    return legal;
  };

  inline void transpose() {
    static constexpr uint64_t diag =
        0b0'1111111'0000000'0000000'0000000'1111111'0000000'0000000'0000000'1111111;
    static constexpr uint64_t bottom =
        0b0'0000000'0000000'0000000'1111111'0000000'0000000'0000000'1111111'0000000;
    static constexpr uint64_t upper =
        0b0'0000000'1111111'0000000'0000000'0000000'1111111'0000000'0000000'0000000;
    static constexpr uint64_t left_corner =
        0b0'0000000'0000000'0000000'0000000'0000000'0000000'1111111'0000000'0000000;
    static constexpr uint64_t right_corner =
        0b0'0000000'0000000'1111111'0000000'0000000'0000000'0000000'0000000'0000000;
    raw = (raw & diag) | ((raw & bottom) << 14) | ((raw & upper) >> 14) |
          ((raw & left_corner) << 28) | ((raw & right_corner) >> 28);
  };
  inline void mirror() {
    static constexpr uint64_t first_col =
        0b0'1111111'0000000'0000000'1111111'0000000'0000000'1111111'0000000'0000000;
    static constexpr uint64_t second_col =
        0b0'0000000'1111111'0000000'0000000'1111111'0000000'0000000'1111111'0000000;
    static constexpr uint64_t third_col =
        0b0'0000000'0000000'1111111'0000000'0000000'1111111'0000000'0000000'1111111;
    raw =
        (raw & second_col) | (raw & first_col) >> 14 | (raw & third_col) << 14;
  }
  inline void flip() {
    static constexpr uint64_t first_row =
        0b0'1111111'1111111'1111111'0000000'0000000'0000000'0000000'0000000'0000000;
    static constexpr uint64_t second_row =
        0b0'0000000'0000000'0000000'1111111'1111111'1111111'0000000'0000000'0000000;
    static constexpr uint64_t third_row =
        0b0'0000000'0000000'0000000'0000000'0000000'0000000'1111111'1111111'1111111;
    raw =
        (raw & second_row) | (raw & first_row) >> 42 | (raw & third_row) << 42;
  }

  inline void rotate_right() {
    transpose();
    mirror();
  }  // clockwise
  inline void rotate_left() {
    transpose();
    flip();
  }  // counterclockwise
  inline void reverse() {
    mirror();
    flip();
  }

  inline uint64_t hash() const {
    auto b = Board(raw);
    uint64_t h = 0xffffffffffffffffull;
    for (int i = 0; i < 4; i++) {
      if (b.raw < h) {
        h = b.raw;
      }
      b.rotate_right();
    }
    b.mirror();
    for (int i = 0; i < 4; i++) {
      if (b.raw < h) {
        h = b.raw;
      }
      b.rotate_right();
    }
    return h;
  }

  std::tuple<Reward, bool> terminated() {
    static constexpr uint64_t bonus_pattern[8] = {
        0b0'1111111'0000000'0000000'1111111'0000000'0000000'1111111'0000000'0000000ULL,  // 3rd col
        0b0'0000000'1111111'0000000'0000000'1111111'0000000'0000000'1111111'0000000ULL,  // 2nd col
        0b0'0000000'0000000'1111111'0000000'0000000'1111111'0000000'0000000'1111111ULL,  // 1st col
        0b0'1111111'1111111'1111111'0000000'0000000'0000000'0000000'0000000'0000000ULL,  // 3rd row
        0b0'0000000'0000000'0000000'1111111'1111111'1111111'0000000'0000000'0000000ULL,  // 2nd row
        0b0'0000000'0000000'0000000'0000000'0000000'0000000'1111111'1111111'1111111ULL,  // 1st row
        0b0'1111111'0000000'0000000'0000000'1111111'0000000'0000000'0000000'1111111ULL,  // diag
        0b0'0000000'0000000'1111111'0000000'1111111'0000000'1111111'0000000'0000000ULL,  // flipped-diag
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
    if (!legal(action)) std::cout << "ILLEGAL!\n";
    auto&& [idxs, minus] = action2idx(action);
    static constexpr uint64_t row_or_col[6] = {
        0b0'0000000'0000000'0000000'0000000'0000000'0000000'0000001'0000001'0000001ULL,  // 1st row
        0b0'0000000'0000000'0000000'0000001'0000001'0000001'0000000'0000000'0000000ULL,  // 2nd row
        0b0'0000001'0000001'0000001'0000000'0000000'0000000'0000000'0000000'0000000ULL,  // 3rd row
        0b0'0000000'0000000'0000001'0000000'0000000'0000001'0000000'0000000'0000001ULL,  // 1st col
        0b0'0000000'0000001'0000000'0000000'0000001'0000000'0000000'0000001'0000000ULL,  // 2nd col
        0b0'0000001'0000000'0000000'0000001'0000000'0000000'0000001'0000000'0000000ULL,  // 3rd col
    };
    raw -= minus * row_or_col[action % 6];
    auto&& [rt, done] = terminated();
    return {rt - minus, done};
  };
  Board& operator=(const Board& b) {
    raw = b.raw;
    return *this;
  };
  int max_min_unit() {
    int max = 0;
    for (const auto& idx : idxs) {
      int min = 100;
      for (const auto& i : idx) {
        min = std::min(get(i), min);
      }
      max = std::max(max, min);
    }
    return max;
  }
  int min_max_unit() {
    int min = 100;
    for (int i = 0; i < 9; i++) min = std::min(get(i), min);
    return min;
  };

  std::vector<Action> shuffle_legal_move(bool heuristic = false, int low = 0,
                                         int N = 18) const {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::vector<int> numbers(N - low, 0);
    std::iota(numbers.begin(), numbers.end(), low);
    if (heuristic) {
      // std::reverse(numbers.begin(), numbers.end());
      for (int i = 0; i < 3; i++) {
        std::shuffle(numbers.begin() + i * 6, numbers.begin() + (i + 1) * 6,
                     gen);
      }
    } else {
      std::shuffle(numbers.begin(), numbers.end(), gen);
    }

    // Shuffle the numbers randomly
    auto view = numbers | std::views::filter(
                              [this](int action) { return legal(action); });
    return {view.begin(), view.end()};
  };

  const std::array<std::tuple<int, int>, 6> min_of_each() const {
    std::array<std::tuple<int, int>, 6> result = {};
    for (int i = 0; i < 6; i++) {
      int min = std::numeric_limits<int>::infinity();
      int min_id = 0;
      for (int j = 0; j < 3; j++) {
        int val = get(idxs[i][j]);
        if (val < min) {
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
  } os << "\n";
  return os;
}

std::istream& operator>>(std::istream& is, Board& b) {
    is >> b.raw;
    return is;
}