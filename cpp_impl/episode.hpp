#pragma once
#include <vector>

#include "agent.hpp"
#include "board.hpp"
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

episode play_an_episode(player& p1, player& p2, const int play_first = 0,
                        int board_max = 99, int board_min = 50,
                        const bool verbose = false) {
  board b;
  for (int i = 0; i < 9; i++) {
    b.set(i, std::rand() % (board_max - board_min + 1) + board_min);
  };

  episode ep;
  int pid = play_first;

  while (true) {
    auto& who = (pid % 2 == 0) ? p1 : p2;
    int action = who.generate(b);
    auto&& [reward, done] = b.apply(action);
    // if (verbose) {
    //   std::cout << b_prev;
    //   std::cout << "Select: " << action % 6 << " Substract: " << action / 6 +
    //   1
    //             << " Reward: " << reward << "\n";
    // }
    ep.add({action, reward, b});
    ep.scores[pid] += reward;
    if (done) break;
    pid = (pid + 1) % 2;
  }
  return ep;
}

std::tuple<float, float> test_player0(player& p1, player& p2, int num_to_play = 100,
                  size_t b_max = 99, size_t b_min = 50, bool verbose=false) {
  int first_win = 0;
  for (int i_episode = 0; i_episode < num_to_play / 2; i_episode++) {
    episode ep = play_an_episode(p1, p2, 0, b_max, b_min);
    if (ep.win() == 0) first_win++;
  }
  if(verbose) std::cout << " | first win rate: " << float(first_win) / (num_to_play/2) << " | ";
  int second_win = 0;
  for (int i_episode = 0; i_episode < num_to_play / 2; i_episode++) {
    episode ep = play_an_episode(p1, p2, 1, b_max, b_min);
    if (ep.win() == 0) second_win++;
  }
  if(verbose) std::cout << "second win rate: " << float(second_win) / (num_to_play/2) << "\n";
  return {float(first_win) / (num_to_play/2), float(second_win) / (num_to_play/2)};
}