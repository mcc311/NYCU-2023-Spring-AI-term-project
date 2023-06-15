#pragma once
#include <numeric>
#include <vector>
#include <time.h>

#include <sstream>
#include <iostream>
#include "agent.hpp"
#include "board.hpp"
class ReplayBuffer;
class Episode {
  friend class ReplayBuffer;
 public:
  typedef struct step{ // At, R_t, St+1
    Board::Action action;
    Board::Reward reward;
    Board state;
    step(Board::Action action=-1, Board::Reward reward=0, Board state={}):
      action(action), reward(reward), state(state){};
  } Step;

  Board init_state;
  Episode(Board init_state=0): init_state(init_state){};
  std::vector<Step> history;
  float scores[2] = {};
  clock_t max_time[2] = {-std::numeric_limits<clock_t>::infinity(),
                       -std::numeric_limits<clock_t>::infinity()};
  clock_t time = 0;
  size_t size() { return history.size(); };
  void add(const Step& s) { history.push_back(s); }
  int win() {
    if (scores[0] > scores[1]) return 0;
    if (scores[0] < scores[1]) return 1;
    return -1;
  };
  void save(const int i, const std::string& filename) {
    std::ofstream out(filename, std::ios::app);
    out << i << " " << scores[0] << " " << scores[1] << " " << time << " " << init_state.raw << " ";
    for (auto&& [action, reward, state] : history) {
      out << action << " " << reward << " " << state.raw << " ";
    }
    out <<  "\n";
    out.close();
  }
};

Episode PlayAnEpisode(player& p1, player& p2, const int play_first = 0,
                      int Board_max = 99, int Board_min = 50,
                      const bool verbose = false) {
  Board b;
  for (int i = 0; i < 9; i++) {
    b.set(i, std::rand() % (Board_max - Board_min + 1) + Board_min);
  };

  Episode ep(b);
  int pid = play_first;
  auto ep_start = clock();

  while (true) {
    auto& who = (pid % 2 == 0) ? p1 : p2;
    auto start = clock();
    int action = who.generate(b);
    auto elapse = clock() - start;
    ep.max_time[pid] = (elapse > ep.max_time[pid]) ? elapse : ep.max_time[pid];
    auto&& [reward, done] = b.apply(action);
    if (verbose) {
      std::cout << b;
      std::cout << "Select: " << action % 6 << " Substract: " << action / 6 + 1
                << " Reward: " << reward << "Takes: " << elapse / CLOCKS_PER_SEC
                << "sec\n";
    }
    ep.add({action, reward, b});
    ep.scores[pid] += reward;
    if (done) break;
    pid = (pid + 1) % 2;
  }
  auto ep_elapse = clock() - ep_start;
  ep.time = ep_elapse / CLOCKS_PER_SEC;
  if (verbose) {
    std::cout << "Player " << ep.win() << " win!\n";
  }
  return ep;
}

std::tuple<float, float> test_player0(player& p1, player& p2,
                                      int num_to_play = 100, size_t b_max = 99,
                                      size_t b_min = 50, bool verbose = false) {
  int first_win = 0;
  std::vector<float> max_time_p1, max_time_p2;
  for (int i_episode = 0; i_episode < num_to_play / 2; i_episode++) {
    Episode ep = PlayAnEpisode(p1, p2, 0, b_max, b_min, verbose);
    max_time_p1.push_back(ep.max_time[0]);
    max_time_p2.push_back(ep.max_time[1]);
    if (ep.win() == 0) first_win++;
  }
  int second_win = 0;
  for (int i_episode = 0; i_episode < num_to_play / 2; i_episode++) {
    Episode ep = PlayAnEpisode(p1, p2, 1, b_max, b_min, verbose);
    max_time_p1.push_back(ep.max_time[0]);
    max_time_p2.push_back(ep.max_time[1]);
    if (ep.win() == 0) second_win++;
  }

  auto mean_and_std = [](std::vector<float> v) -> std::tuple<float, float> {
    float mean = std::reduce(v.cbegin(), v.cend()) / v.size();
    float stddev =
        std::transform_reduce(v.cbegin(), v.cend(), 0.0f, std::plus{},
                              [&mean](auto val) -> float {
                                return (val - mean) * (val - mean);
                              }) /
        v.size();
    return {mean, std::sqrt(stddev)};
  };
  auto&& [p1_mean, p1_std] = mean_and_std(max_time_p1);
  auto&& [p2_mean, p2_std] = mean_and_std(max_time_p2);
  std::cout << " | first win rate: " << float(first_win) / (num_to_play / 2)
            << " | second win rate: " << float(second_win) / (num_to_play / 2)
            << std::endl;
  std::cout << "P1 cost: " << p1_mean / CLOCKS_PER_SEC << "±"
            << p1_std / CLOCKS_PER_SEC << "secs/step." << std::endl;
  std::cout << "P2 cost: " << p2_mean / CLOCKS_PER_SEC << "±"
            << p2_std / CLOCKS_PER_SEC << "secs/step." << std::endl;

  return {float(first_win) / (num_to_play / 2),
          float(second_win) / (num_to_play / 2)};
}

std::ostream& operator<<(std::ostream& os, const Episode& ep) {
  os << "Player 0: " << ep.scores[0] << " Player 1: " << ep.scores[1]
     << " Time: " << ep.time << "sec\n";
  os << ep.init_state;
  for (auto&& [action, reward, state] : ep.history) {
    os << action << " " << reward << "\n" << state << " ";
  } os << std::endl;
  return os;
}

class ReplayBuffer {
 public:

  std::vector<Episode> buffer;
  size_t capacity;
  ReplayBuffer(size_t capacity) : capacity(capacity){
    buffer.reserve(capacity);
  };
  void push(const Episode& s) {
    if (buffer.size() >= capacity) {
      buffer.erase(buffer.begin());
    }
    buffer.push_back(s);
  }
  Episode sample() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dist(0, buffer.size() - 1);
    return buffer[dist(gen)];
  }
  size_t size() { return buffer.size(); }

  
  void load(const std::string& filename) {
    std::ifstream in(filename);
    std::string line;
    while (std::getline(in, line)) {
      std::stringstream ss(line);
      Episode ep;
      int id;
      ss >> id;
      ss >> ep.scores[0] >> ep.scores[1] >> ep.time;
      ss >> ep.init_state;
      while (true) {
        int action, reward;
        Board state;
        Episode::Step step;
        ss >> action >> reward >> state;
        step.action = action;
        step.reward = reward;
        step.state = state;
        if (ss.eof()) break;
        ep.history.push_back(step);
        
      }
      push(ep);
    }
    in.close();
  };

};