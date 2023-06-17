#pragma once
#include <sstream>
#include <vector>

#include "board.hpp"
#include "episode.hpp"
class ReplayBuffer {
 public:
  std::vector<Episode> buffer;
  size_t capacity;
  ReplayBuffer(size_t capacity) : capacity(capacity) {
    buffer.reserve(capacity);
  };
  ReplayBuffer(std::vector<Episode> buffer)
      : buffer(buffer), capacity(buffer.size()){};
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
  void shuffle() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::shuffle(buffer.begin(), buffer.end(), gen);
  };
  std::tuple<ReplayBuffer, ReplayBuffer> split(const float ratio) {
    ReplayBuffer buffer1(
        {buffer.begin(), buffer.begin() + size_t(capacity * ratio)});
    ReplayBuffer buffer2(
        {buffer.begin() + size_t(capacity * ratio), buffer.end()});
    return {buffer1, buffer2};
  };
  // overload iterator to iterate over buffer
  std::vector<Episode>::iterator begin() { return buffer.begin(); };
  std::vector<Episode>::iterator end() { return buffer.end(); };
};