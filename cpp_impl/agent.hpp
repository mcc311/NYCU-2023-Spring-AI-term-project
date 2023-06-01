#pragma once
#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <random>
#include <ranges>
#include <vector>

#include "board.hpp"
#include "utils.hpp"

class player {
 protected:
  static constexpr int feat_num = 5;
  static constexpr int feat_size = 3;
  static constexpr int isom_num = 8;
  // static constexpr int feat_idxs[feat_num][feat_size] = {
  //     {0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {0, 3, 6},
  //     {1, 4, 7}, {2, 5, 8}, {0, 4, 8}, {2, 4, 6}};
  // static constexpr int feat_idxs[feat_num][feat_size] = {
  //     {0, 1, 2, 3}, {3, 4, 5, 6}, {3, 4, 6, 7}, {0, 4, 8, 7}};
  static constexpr int feat_idxs[feat_num][feat_size] = {
      {0, 1, 2}, {3, 4, 5}, {0, 4, 8}, {0, 1, 3}, {3, 4, 7}};
  inline static int net_size = int(pow(100, feat_size));
  std::vector<std::vector<float> > net;

 public:
  player() : net(feat_num, std::vector<float>(net_size, 0)){};
  virtual float evaluate(const board& b) { return 0; };
  virtual int generate(board& b) { return 0; };
  void update(const board& b, const float error, const float alpha = 0.001) {
    for (int i = 0; i < feat_num; i++) {
      for (int isom = 0; isom < isom_num; isom++) {
        uint32_t feat = 0;
        for (const auto& f : feat_idxs[i]) {
          feat *= 100;
          feat += b.get(f, isom);
        }
        net[i][feat] += alpha * error / (feat_num * isom_num);
      }
    }
  };
  player& operator=(const player& who) {
    for (int i = 0; i < feat_num; i++) {
      std::copy(who.net[i].begin(), who.net[i].end(), net[i].begin());
    }
    return *this;
  }

  void save(const std::string& path) {
    for (int i = 0; i < feat_num; i++) {
      std::ofstream out;
      out.open((path + "/" + std::to_string(i) + ".bin").c_str(),
               std::ios::out | std::ios::binary | std::ios::trunc);
      if (out.is_open()) {
        out.write((char*)(net[i].data()), net[i].size());
        out.flush();
        out.close();
      }
    }
  };
  void load(const std::string& path) {
    for (int i = 0; i < feat_num; i++) {
      std::ifstream in;
      in.open((path + "/" + std::to_string(i) + ".bin").c_str(),
              std::ios::in | std::ios::binary);
      if (in.is_open()) {
        in.read((char*)(net[i].data()), net[i].size());
        in.close();
      }
    }
  };
};


class td_player : public player {
 public:
  td_player(){};
  virtual float evaluate(const board& b) {
    float value = 0;
    for (int i = 0; i < feat_num; i++) {
      for (int isom = 0; isom < isom_num; isom++) {
        uint32_t feat = 0;
        for (const auto& f : feat_idxs[i]) {
          feat = feat * 100;
          feat += b.get(f, isom);
        }
        value += net[i][feat];
      }
    }
    return value / (feat_num * isom_num);
  }
  virtual int generate(board& b) {
    int best_action = -1;
    float best_value = -MAXFLOAT;
    for (auto& action :
         shuffle_actions() |
             std::views::filter([&b](int action) { return b.legal(action); })) {
      auto b_ = b;
      // if (!b_.legal(action)) continue;
      auto&& [reward, done] = b_.apply(action);
      auto value = reward - evaluate(b_);  // for two td_player
      if (value > best_value) {
        best_value = value;
        best_action = action;
      }
    }
    return best_action;
  }
};

class ab_player : public player {
 public:
  ab_player(size_t max_depth = 6) : max_depth(max_depth){};
  virtual float evaluate(const board& b) {
    float value = 0;
    for (int i = 0; i < feat_num; i++) {
      for (int isom = 0; isom < isom_num; isom++) {
        uint32_t feat = 0;
        for (const auto& f : feat_idxs[i]) {
          feat = feat * 100;
          feat += b.get(f, isom);
        }
        value += net[i][feat];
      }
    }
    return value / (feat_num * isom_num);
  };

  board::reward alphaBetaMin(board, board::reward, board::reward, int);
  board::reward alphaBetaMax(board, board::reward, board::reward, int);

  virtual int generate(board& b) {
    int best_action = -1;
    float best_value = -MAXFLOAT;
    // for (auto& action :
    //      shuffle_actions()
    for (auto& action :
         shuffle_actions() |
             std::views::filter([&b](int action) { return b.legal(action); })) {
      auto b_ = b;
      // if (!b_.legal(action)) continue;
      auto&& [reward, done] = b_.apply(action);
      auto value = reward - alphaBetaMax(b_, -MAXFLOAT, MAXFLOAT, max_depth);
      if (value > best_value) {
        best_value = value;
        best_action = action;
      }
    }
    return best_action;
  }

 private:
  size_t max_depth;
  static constexpr int threshold = 20;
};

board::reward ab_player::alphaBetaMax(board b, board::reward alpha,
                                      board::reward beta, int depthleft) {
  if (depthleft == 0)
    return evaluate(b);  // Reach end depth or terminal condition.
  for (const auto& action :
       shuffle_actions()  // std::views::iota(18)
           | std::views::filter([&b](int action) { return b.legal(action); })) {
    board b_ = b;

    auto&& [r, done] = b_.apply(action);
    int score = r + (done ? 0 : alphaBetaMin(b_, alpha, beta, depthleft - 1));
    // int score = r + (alphaBetaMin(b_, alpha, beta, done ? 0 : depthleft - 1));
    if (score >= beta) return beta;    // fail hard beta-cutoff
    if (score > alpha) alpha = score;  // alpha acts like max in MiniMax
  }
  return alpha;
};

board::reward ab_player::alphaBetaMin(board b, board::reward alpha,
                                      board::reward beta, int depthleft) {
  if (depthleft == 0) return -evaluate(b);
  for (const auto& action :
       shuffle_actions()  // std::views::iota(18)
                          // | std::views::reverse
           | std::views::filter([&b](int action) { return b.legal(action); })) {
    board b_ = b;
    auto&& [r, done] = b_.apply(action);
    int score = -r +  (done ? 0 : alphaBetaMax(b_, alpha, beta, depthleft - 1));
    // int score = -r +  (alphaBetaMax(b_, alpha, beta, done ? 0 :  depthleft - 1));
    if (score <= alpha) return alpha;  // fail hard alpha-cutoff
    if (score < beta) beta = score;    // beta acts like min in MiniMax
  }
  return beta;
};


class heuristic_ab_player : public player {
 public:
  heuristic_ab_player(size_t max_depth = 6) : max_depth(max_depth){};
  virtual float evaluate(const board& b) {
    float value = 0;
    for (int i = 0; i < feat_num; i++) {
      for (int isom = 0; isom < isom_num; isom++) {
        uint32_t feat = 0;
        for (const auto& f : feat_idxs[i]) {
          feat = feat * 100;
          feat += b.get(f, isom);
        }
        value += net[i][feat];
      }
    }
    return value / (feat_num * isom_num);
  };

  board::reward alphaBetaMin(board, board::reward, board::reward, int);
  board::reward alphaBetaMax(board, board::reward, board::reward, int);

  virtual int generate(board& b) {
    int best_action = -1;
    float best_value = -MAXFLOAT;
    // for (auto& action :
    //      shuffle_actions()
    for (auto& action :
         shuffle_actions() |
             std::views::filter([&b](int action) { return b.legal(action); })) {
      auto b_ = b;
      // if (!b_.legal(action)) continue;
      auto&& [reward, done] = b_.apply(action);
      auto value = reward - alphaBetaMax(b_, -MAXFLOAT, MAXFLOAT, max_depth);
      if (value > best_value) {
        best_value = value;
        best_action = action;
      }
    }
    return best_action;
  }

 private:
  size_t max_depth;
  static constexpr int threshold = 20;
};

board::reward heuristic_ab_player::alphaBetaMax(board b, board::reward alpha,
                                      board::reward beta, int depthleft) {
  if (depthleft == 0)
    return evaluate(b);  // Reach end depth or terminal condition.
  int low = b.max_min_unit() > threshold ?  12 :  0;
  int N   = b.max_min_unit() > threshold ?  18 :  18;
  for (const auto& action :
       shuffle_actions(low, N)  // std::views::iota(18)
           | std::views::filter([&b](int action) { return b.legal(action); })) {
    board b_ = b;

    auto&& [r, done] = b_.apply(action);
    int score = r + (done ? 0 : alphaBetaMin(b_, alpha, beta, depthleft - 1));
    // int score = r + (alphaBetaMin(b_, alpha, beta, done ? 0 : depthleft - 1));
    if (score >= beta) return beta;    // fail hard beta-cutoff
    if (score > alpha) alpha = score;  // alpha acts like max in MiniMax
  }
  return alpha;
};

board::reward heuristic_ab_player::alphaBetaMin(board b, board::reward alpha,
                                      board::reward beta, int depthleft) {
  if (depthleft == 0) return -evaluate(b);
  int low = b.max_min_unit() > threshold ?  12 :  0;
  int N   = b.max_min_unit() > threshold ?  18 :  18;
  for (const auto& action :
       shuffle_actions(low, N)  // std::views::iota(18)
                          // | std::views::reverse
           | std::views::filter([&b](int action) { return b.legal(action); })) {
    board b_ = b;
    auto&& [r, done] = b_.apply(action);
    int score = -r +  (done ? 0 : alphaBetaMax(b_, alpha, beta, depthleft - 1));
    // int score = -r +  (alphaBetaMax(b_, alpha, beta, done ? 0 :  depthleft - 1));
    if (score <= alpha) return alpha;  // fail hard alpha-cutoff
    if (score < beta) beta = score;    // beta acts like min in MiniMax
  }
  return beta;
};



class heuristic_player : public player {
 public:
  heuristic_player(){};
  virtual int generate(board& b) {
    
  }
};