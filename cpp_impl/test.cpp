#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <random>
#include <ranges>
#include <vector>

#include "agent.hpp"
#include "board.hpp"
#include "episode.hpp"
#include "utils.hpp"


int main(int argc, const char* argv[]) {
  std::copy(argv, argv + argc,
            std::ostream_iterator<const char*>(std::cout, " "));
  std::cout << std::endl;
  size_t total = 1000000, block = 10000;
  float alpha = 0.01;
  size_t b_max = 99, b_min = 50, max_depth = 3, test_num = 100;
  std::string slide_args, place_args;
  std::string load_path = "", save_path = "";
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    auto match_arg = [&](std::string flag) -> bool {
      auto it = arg.find_first_not_of('-');
      return arg.find(flag, it) == it;
    };
    auto next_opt = [&]() -> std::string {
      auto it = arg.find('=') + 1;
      return it ? arg.substr(it) : argv[++i];
    };
    if (match_arg("total")) {
      total = std::stoull(next_opt());
    } else if (match_arg("alpha")) {
      alpha = std::stof(next_opt());
    } else if (match_arg("block")) {
      block = std::stoull(next_opt());
    } else if (match_arg("load")) {
      load_path = next_opt();
    } else if (match_arg("save")) {
      save_path = next_opt();
    } else if (match_arg("b_max")) {
      b_max = std::stoull(next_opt());
    } else if (match_arg("b_min")) {
      b_min = std::stoull(next_opt());
    } else if (match_arg("max_depth")) {
      max_depth = std::stoull(next_opt());
    } else if (match_arg("test_num")) {
      test_num = std::stoull(next_opt());
    }
  }
  // static std::vector<Episode> replay_buffer;
  // replay_buffer.reserve(1000);
  // auto p1 = ab_player(max_depth);
  // p1.load("td-8M-ac-10k");
  // auto p2 = heuristic_ab_player(max_depth);
  // p2.load("td-8M-ac-10k");
  auto p1 = nega_player(10, false);
  auto p2 = pvs_player(10, false);
  auto&& [first_win_rate, second_win_rate] =
          test_player0((player&)p1, (player&)p2, 10, 99, 50, false);

  // Test();
}