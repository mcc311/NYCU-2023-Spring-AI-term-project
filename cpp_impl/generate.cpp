#include <iostream>

#include "agent.hpp"
#include "board.hpp"
#include "episode.hpp"
#include "utils.hpp"

// use `mcts_player` generate `episode`, and save them in file, with format like
// this: state(uint64_t) action(int) state(uint64_t) action(int) ...
int main(int argc, const char* argv[]) {
  std::copy(argv, argv + argc,
            std::ostream_iterator<const char*>(std::cout, " "));
  std::cout << std::endl;
  size_t total = 1'000'000, block = 10000;
  size_t sim_count = 10000;
  std::string id = "";

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
    } else if (match_arg("sim_count")) {
      sim_count = std::stof(next_opt());
    } else if (match_arg("block")) {
      block = std::stoull(next_opt());
    } else if (match_arg("id")) {
      id = next_opt();
    }
  }
  std::string save_path = "trajectory/" + id + "_" + std::to_string(total / 1000000) + "M" +
                          std::to_string(total / 1000) + "k" +
                          std::to_string(sim_count) + "sim.episode";
  for (int i = 1; i < total; i++) {
    if (i % block == 0) {
      std::cout << "block " << i / block << std::endl;
    }
    auto p1 = mcts_player(sim_count);
    auto p2 = mcts_player(sim_count);
    auto ep = PlayAnEpisode(p1, p2);
    ep.save(i, save_path);
  }
}