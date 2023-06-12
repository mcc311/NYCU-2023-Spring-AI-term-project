#pragma once
#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <ranges>
#include <vector>

#include "board.hpp"
#include "mcts.hpp"
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
  std::vector<std::vector<float> > n_tuple_net;

 public:
  player() : n_tuple_net(feat_num, std::vector<float>(net_size, 0)){};
  virtual float evaluate(const Board& b) { return 0; };
  virtual Board::Action generate(Board& b) { return 0; };
  void update(const Board& b, const float error, const float alpha = 0.001) {
    for (int i = 0; i < feat_num; i++) {
      for (int isom = 0; isom < isom_num; isom++) {
        uint32_t feat = 0;
        for (const auto& f : feat_idxs[i]) {
          feat *= 100;
          feat += b.get(f, isom);
        }
        n_tuple_net[i][feat] += alpha * error / (feat_num * isom_num);
      }
    }
  };
  virtual player& operator=(const player& who) {
    for (int i = 0; i < feat_num; i++) {
      std::copy(who.n_tuple_net[i].begin(), who.n_tuple_net[i].end(),
                n_tuple_net[i].begin());
    }
    return *this;
  }

  virtual void save(const std::string& path) {
    for (int i = 0; i < feat_num; i++) {
      std::ofstream out;
      out.open((path + "/" + std::to_string(i) + ".bin").c_str(),
               std::ios::out | std::ios::binary | std::ios::trunc);
      if (out.is_open()) {
        out.write((char*)(n_tuple_net[i].data()), n_tuple_net[i].size());
        out.flush();
        out.close();
      }
    }
  };
  virtual void load(const std::string& path) {
    for (int i = 0; i < feat_num; i++) {
      std::ifstream in;
      in.open((path + "/" + std::to_string(i) + ".bin").c_str(),
              std::ios::in | std::ios::binary);
      if (in.is_open()) {
        in.read((char*)(n_tuple_net[i].data()), n_tuple_net[i].size());
        in.close();
      }
    }
  };
};

class td_player : public player {
 public:
  td_player(){};
  virtual float evaluate(const Board& b) {
    float value = 0;
    for (int i = 0; i < feat_num; i++) {
      for (int isom = 0; isom < isom_num; isom++) {
        uint32_t feat = 0;
        for (const auto& f : feat_idxs[i]) {
          feat = feat * 100;
          feat += b.get(f, isom);
        }
        value += n_tuple_net[i][feat];
      }
    }
    return value / (feat_num * isom_num);
  }
  virtual Board::Action generate(Board& b) {
    int best_action = -1;
    float best_value = -std::numeric_limits<float>::infinity();
    for (auto& action : b.shuffle_legal_move()) {
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

class ab_player : public td_player {
 public:
  ab_player(size_t max_depth = 6) : max_depth(max_depth){};
  Board::Reward alphaBetaMin(Board, Board::Reward, Board::Reward, int);
  Board::Reward alphaBetaMax(Board, Board::Reward, Board::Reward, int);

  virtual Board::Action generate(Board& b) {
    int best_action = -1;
    float best_value = -std::numeric_limits<float>::infinity();
    // for (auto& action :
    //      ShuffleActions()
    for (auto& action : b.shuffle_legal_move()) {
      auto b_ = b;
      // if (!b_.legal(action)) continue;
      auto&& [reward, done] = b_.apply(action);
      auto value =
          reward - alphaBetaMax(b_, -std::numeric_limits<float>::infinity(),
                                std::numeric_limits<float>::infinity(),
                                max_depth);
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

Board::Reward ab_player::alphaBetaMax(Board b, Board::Reward alpha,
                                      Board::Reward beta, int depthleft) {
  if (depthleft == 0)
    return evaluate(b);  // Reach end depth or terminal condition.
  for (const auto& action : b.shuffle_legal_move()) {
    Board b_ = b;

    auto&& [r, done] = b_.apply(action);
    int score = r + (done ? 0 : alphaBetaMin(b_, alpha, beta, depthleft - 1));
    // int score = r + (alphaBetaMin(b_, alpha, beta, done ? 0 : depthleft -
    // 1));
    if (score >= beta) return beta;    // fail hard beta-cutoff
    if (score > alpha) alpha = score;  // alpha acts like max in MiniMax
  }
  return alpha;
};

Board::Reward ab_player::alphaBetaMin(Board b, Board::Reward alpha,
                                      Board::Reward beta, int depthleft) {
  if (depthleft == 0) return -evaluate(b);
  for (const auto& action : b.shuffle_legal_move()) {
    Board b_ = b;
    auto&& [r, done] = b_.apply(action);
    int score = -r + (done ? 0 : alphaBetaMax(b_, alpha, beta, depthleft - 1));
    // int score = -r +  (alphaBetaMax(b_, alpha, beta, done ? 0 :  depthleft -
    // 1));
    if (score <= alpha) return alpha;  // fail hard alpha-cutoff
    if (score < beta) beta = score;    // beta acts like min in MiniMax
  }
  return beta;
};

class heuristic_ab_player : public player {
 public:
  heuristic_ab_player(size_t max_depth = 6) : max_depth(max_depth){};
  virtual float evaluate(const Board& b) {
    float value = 0;
    for (int i = 0; i < feat_num; i++) {
      for (int isom = 0; isom < isom_num; isom++) {
        uint32_t feat = 0;
        for (const auto& f : feat_idxs[i]) {
          feat = feat * 100;
          feat += b.get(f, isom);
        }
        value += n_tuple_net[i][feat];
      }
    }
    return value / (feat_num * isom_num);
  };

  Board::Reward alphaBetaMin(Board b, Board::Reward alpha, Board::Reward beta,
                             int depthleft) {
    if (depthleft == 0) return -evaluate(b);
    int low = b.max_min_unit() > threshold ? 12 : 0;
    int N = b.max_min_unit() > threshold ? 18 : 18;
    for (const auto& action : b.shuffle_legal_move(low, N)) {
      Board b_ = b;
      auto&& [r, done] = b_.apply(action);
      int score =
          -r + (done ? 0 : alphaBetaMax(b_, alpha, beta, depthleft - 1));
      // int score = -r +  (alphaBetaMax(b_, alpha, beta, done ? 0 :  depthleft
      // - 1));
      if (score <= alpha) return alpha;  // fail hard alpha-cutoff
      if (score < beta) beta = score;    // beta acts like min in MiniMax
    }
    return beta;
  };
  Board::Reward alphaBetaMax(Board b, Board::Reward alpha, Board::Reward beta,
                             int depthleft) {
    if (depthleft == 0)
      return evaluate(b);  // Reach end depth or terminal condition.
    int low = b.max_min_unit() > threshold ? 12 : 0;
    int N = b.max_min_unit() > threshold ? 18 : 18;
    for (const auto& action : b.shuffle_legal_move(low, N)) {
      Board b_ = b;

      auto&& [r, done] = b_.apply(action);
      int score = r + (done ? 0 : alphaBetaMin(b_, alpha, beta, depthleft - 1));
      // int score = r + (alphaBetaMin(b_, alpha, beta, done ? 0 : depthleft -
      // 1));
      if (score >= beta) return beta;    // fail hard beta-cutoff
      if (score > alpha) alpha = score;  // alpha acts like max in MiniMax
    }
    return alpha;
  };

  virtual Board::Action generate(Board& b) {
    int best_action = -1;
    float best_value = -std::numeric_limits<float>::infinity();
    // for (auto& action :
    //      ShuffleActions()
    for (auto& action : b.shuffle_legal_move()) {
      auto b_ = b;
      // if (!b_.legal(action)) continue;
      auto&& [reward, done] = b_.apply(action);
      auto value =
          reward - alphaBetaMax(b_, -std::numeric_limits<float>::infinity(),
                                std::numeric_limits<float>::infinity(),
                                max_depth);
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

class mcts_player : public player {
 public:
  int sim_count;
  mcts_player(int sim_count = 500) : sim_count(sim_count){};
  virtual Board::Action generate(Board& b) {
    return monte_carlo_tree_search(b, sim_count);
  }
};

class random_player : public player {
 public:
  random_player(){};
  virtual Board::Action generate(Board& b) { return b.shuffle_legal_move()[0]; }
};

class nega_player : public player {
 public:
  int max_depth;
  nega_player(int max_depth = 3) : max_depth(max_depth){};
  float negamaxSearch(const Board& b, int depth, bool done, float alpha,
                      float beta) {
    // // Check if the state has already been evaluated
    // long long hashValue = calculateHash(state);
    // if (transpositionTable.count(hashValue) > 0) {
    //   return transpositionTable[hashValue];
    // }

    // Check if the search has reached the maximum depth or the game is over
    if (depth == 0 || done) {
      return evaluate(b);
    }

    float maxEval = -std::numeric_limits<float>::infinity();
    // ... generate possible moves and evaluate them
    for (const Board::Action& action : b.shuffle_legal_move()) {
      Board b_ = b;
      auto&& [r, done] = b_.apply(action);
      float eval = r - negamaxSearch(b_, depth - 1, done, -beta, -alpha);
      maxEval = std::max(maxEval, eval);
      alpha = std::max(alpha, eval);
      if (beta <= alpha) {
        break;  // Beta cutoff
      }
    }
    // Store the evaluated state in the transposition table
    // transpositionTable[hashValue] = maxEval;
    return maxEval;
  };

  // Function to initiate the Negamax Alpha-Beta pruning search
  float negamaxAlphaBeta(const Board& state, int depth) {
    // Clear the transposition table

    // Call the Negamax search with initial alpha and beta values
    return negamaxSearch(state, depth, false,
                         -std::numeric_limits<float>::infinity(),
                         std::numeric_limits<float>::infinity());
  }
  virtual Board::Action generate(Board& b) {
    if (b.max_min_unit() <= 18) {
      int best_action = -1;
      float best_value = -std::numeric_limits<float>::infinity();
      // for (auto& action :
      //      ShuffleActions()
      for (auto& action : b.shuffle_legal_move()) {
        auto b_ = b;
        // if (!b_.legal(action)) continue;
        auto&& [reward, done] = b_.apply(action);
        auto value = reward - negamaxAlphaBeta(b_, max_depth);
        if (value > best_value) {
          best_value = value;
          best_action = action;
        }
      }
      return best_action;
    }
    return std::rand() % 6;
  };
};

class heuristic_player : public nega_player {
 public:
  std::array<float, 300 * 300 * 300>* net;
  heuristic_player(int max_depth) : nega_player(max_depth){
    net = new std::array<float, 300 * 300 * 300>();
  };
  ~heuristic_player() { delete net; }
  virtual float evaluate(const Board& b) {
    float val = 0;
    auto min_of_each = b.min_of_each();
    for (int init : {0, 3}) {
      int feat = 0;
      for (int i = init; i < 3; i++) {
        feat *= 300;
        auto& [min_id, min] = min_of_each[i];
        feat += min + min_id * 100;
      }
      val += (*net)[feat];
    }
    return val;
  }
  virtual void save(const std::string& path) {
    std::ofstream out;
    out.open((path + "/weight.bin").c_str(),
             std::ios::out | std::ios::binary | std::ios::trunc);
    if (out.is_open()) {
      out.write((char*)(net->data()), net->size());
      out.flush();
      out.close();
    }
  };
  virtual void load(const std::string& path) {
    std::ifstream in;
    in.open((path + "/weight.bin").c_str(), std::ios::in | std::ios::binary);
    if (in.is_open()) {
      in.read((char*)(net->data()), net->size());
      in.close();
    }
  };
};