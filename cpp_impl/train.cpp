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

void td_learning(player& p1, player& p2, const std::string& load_path,
                 const std::string& save_path, const float alpha,
                 const int b_max, const int b_min, const size_t total,
                 const size_t block, const size_t max_depth,
                 const size_t test_num) {
  static constexpr size_t max_ep_in_rb = 1;
  int sim_count = 2500;

  std::vector<Episode> replay_buffer;
  clock_t begin_time = clock();
  for (int i_episode = 0; i_episode <= total; i_episode++) {
    p2 = p1;
    if (i_episode % block == 0) {
      float elapse = float(clock() - begin_time) / CLOCKS_PER_SEC;
      std::cout << "Episode " << i_episode << " ( " << elapse << " sec) || vs. MCTS(" << sim_count << ") ";
      auto test_player = mcts_player(sim_count);
      auto&& [first, second] = test_player0((player&)p1, (player&)(test_player), test_num, b_max,
                   b_min, true);
      if((first + second)/2 >= 0.7) {
        sim_count += 500;
      }
      begin_time = clock();
    }
    
    if (replay_buffer.size() >= max_ep_in_rb)
      replay_buffer.erase(replay_buffer.begin());
    replay_buffer.emplace_back(
        PlayAnEpisode((player&)(p1), (player&)(p2), 0, b_max, b_min));

    // TD learning
    float target = 0;
    for (auto& ep : replay_buffer) {
      for (; ep.history.size(); ep.history.pop_back()) {
        const auto& [action, reward, b_next] = ep.history.back();
        float error = target - p1.evaluate(b_next);
        p1.update(b_next, error, alpha);
        target = reward - p1.evaluate(b_next);
      }
    }
  }
  
};

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
  // auto p1 = heuristic_ab_player(max_depth);
  // auto p2 = heuristic_ab_player(max_depth);
  // if (!load_path.empty()) p1.load(load_path);

  auto p1 = td_player();
  auto p2 = td_player();
  // auto p1 = heuristic_player(max_depth);
  // auto p2 = heuristic_player(max_depth);
  td_learning((player&)(p1), (player&)(p2), load_path, save_path, alpha, b_max,
              b_min, total, block, max_depth, test_num);
  p1.save(save_path);
}