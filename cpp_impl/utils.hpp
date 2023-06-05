#pragma once
#include <algorithm>
#include <random>
#include <ranges>
#include <vector>

#include "board.hpp"

auto shuffle_actions = [](int low = 0, int N = 18) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::vector<int> numbers(N-low);
  std::iota(numbers.begin(), numbers.end(), low);

  // Shuffle the numbers randomly
  std::ranges::shuffle(numbers, gen);
  return numbers;
};

void test() {
  board init_b;
  for (int i = 0; i < 9; i++) {
    init_b.set(i, 10);
  }
  std::cout << init_b;
  for (int action = 0; action < 18; action++) {
    board b;
    for (int i = 0; i < 9; i++) {
      b.set(i, 10);
    }
    b.apply(action);
    std::cout << "Select: " << action % 6 << " Substract: " << action / 6 + 1
              << "\n";
    std::cout << b;
  }
  board b = init_b;
  std::cout << "Operator= overloading test: \n";
  std::cout << "Init:\n" << init_b << "Assigned to:\n" << b;
  for (int action = 0; action < 18; action++) {
    board b;
    b = init_b;
    b.apply(action);
    std::cout << "Select: " << action % 6 << " Substract: " << action / 6 + 1
              << "\n";
    std::cout << b;
  }

  std::cout << "\nTermination Test:\n";
  int test_pattern[9][3] = {{0, 1, 2}, {3, 4, 5}, {6, 7, 8},
                            {0, 3, 6}, {1, 4, 7}, {2, 5, 8},
                            {0, 4, 8}, {2, 4, 6}, {1, 5, 6}};
  for (auto& pattern : test_pattern) {
    b = init_b;
    b.set(pattern[0], 0);
    b.set(pattern[1], 0);
    b.set(pattern[2], 0);
    std::cout << b;
    auto&& [r, t] = b.terminated();
    if (t)
      std::cout << "Terminated with reward: " << r << std::endl;
    else
      std::cout << "Not terminated. Reward: " << r << std::endl;
  }
  b = init_b;
  const clock_t begin_time = clock();
  float elapse = float(clock() - begin_time) / CLOCKS_PER_SEC;
};
