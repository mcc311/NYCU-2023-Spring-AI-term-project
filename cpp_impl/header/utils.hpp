#pragma once
#include <algorithm>
#include <random>
#include <ranges>
#include <vector>
#include "board.h"


std::vector<int> shuffle_actions(int N=18) {
  std::vector<int> numbers(N);
  std::iota(numbers.begin(), numbers.end(), 0);

  // Shuffle the numbers randomly
  std::random_device rd;
  std::mt19937 gen(rd());
  std::ranges::shuffle(numbers, gen);
  return numbers;
}

class episode {
 public:
  using step = std::tuple<board::action, board::reward,
                          board>;  // St, At, R_t, St+1
  std::vector<step> history;
  float scores[2] = {};
  void add(const step& s) { history.push_back(s); }
  int win() {
    if (scores[0] > scores[1]) return 0;
    if (scores[0] < scores[1]) return 1;
    return -1;
  }
};