#pragma once
#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <ranges>
#include <thread>
#include <unordered_map>
#include <vector>

#include "board.hpp"
#include "mcts.hpp"
#include "net.hpp"
#include "utils.hpp"

class player {
 public:
  player(){};
  virtual Board::Action generate(Board& b) { return 0; };
};

class random_player : public player {
 public:
  random_player(){};
  virtual Board::Action generate(Board& b) { return b.shuffle_legal_move()[0]; }
};

class mcts_player : public player {
 private:
  bool heuristic;

 public:
  int sim_count;
  mcts_player(int sim_count = 500, bool heuristic = false)
      : sim_count(sim_count), heuristic(heuristic){};
  virtual Board::Action generate(Board& b) {
    return monte_carlo_tree_search(b, sim_count);
  }
};

class nega_player : public player {
 public:
  int max_depth;
  bool heuristic;

  // transposition table
  std::unordered_map<Board::Hash, Board::Reward> transposition_table;
  nega_player(int max_depth = 3, bool heuristic = false)
      : max_depth(max_depth), heuristic(heuristic){};

  Board::Reward evaluate(const Board& b) {
    // return net.evaluate(b);
    return mcts_estimate(b, 100);
  }
  Board::Reward negamaxSearch(const Board& b, int depth, bool done,
                              Board::Reward alpha, Board::Reward beta) {
    // Check if the state has already been evaluated
    Board::Hash hash = b.hash();
    if (transposition_table.count(hash) > 0) {
      return transposition_table[hash];
    }

    // Check if the search has reached the maximum depth or the game is over
    if (done) {
      return 0;
    }

    Board::Reward best_value = -std::numeric_limits<Board::Reward>::infinity();
    // ... generate possible moves and evaluate them
    for (const Board::Action& action : b.shuffle_legal_move(heuristic)) {
      Board b_ = b;
      auto&& [r, done] = b_.apply(action);
      Board::Reward eval =
          r - negamaxSearch(b_, depth - 1, done, -beta, -alpha);
      best_value = std::max(best_value, eval);
      alpha = std::max(alpha, eval);
      if (beta <= alpha) {
        break;  // Beta cutoff
      }
    }
    // Store the evaluated state in the transposition table
    transposition_table[hash] = best_value;
    return best_value;
  };

  virtual Board::Action generate(Board& b) override {
    int best_action = -1;
    Board::Reward best_value = -std::numeric_limits<Board::Reward>::infinity();
    for (auto& action : b.shuffle_legal_move()) {
      auto b_ = b;
      auto&& [reward, done] = b_.apply(action);
      Board::Reward value =
          reward -
          negamaxSearch(b_, max_depth, done,
                        -std::numeric_limits<Board::Reward>::infinity(),
                        std::numeric_limits<Board::Reward>::infinity());
      if (value > best_value) {
        best_value = value;
        best_action = action;
      }
    }
    return best_action;
  };
};

class pvs_player : public nega_player {
 public:
  // transposition table
  pvs_player(int max_depth = 3, bool heuristic = false)
      : nega_player(max_depth, heuristic){};

  Board::Reward principalVariationSearch(const Board& b, int depth, bool done,
                                         Board::Reward alpha,
                                         Board::Reward beta) {
    // Check if the state has already been evaluated
    Board::Hash hash = b.hash();
    if (transposition_table.count(hash) > 0) {
      return transposition_table[hash];
    }

    // Check if the search has reached the maximum depth or the game is over
    if (done) {
      return 0;
    }

    Board::Reward best_value = -std::numeric_limits<Board::Reward>::infinity();
    bool firstChild = true;

    // Generate possible moves and evaluate them
    for (const Board::Action& action : b.shuffle_legal_move(heuristic)) {
      Board b_ = b;
      auto&& [r, done] = b_.apply(action);

      Board::Reward eval;
      if (firstChild) {
        eval = r - principalVariationSearch(b_, depth - 1, done, -beta, -alpha);
        firstChild = false;
      } else {
        eval = r - principalVariationSearch(b_, depth - 1, done, -alpha - 1,
                                            -alpha);
        if (eval > alpha && eval < beta) {
          eval =
              r - principalVariationSearch(b_, depth - 1, done, -beta, -eval);
        }
      }

      best_value = std::max(best_value, eval);
      alpha = std::max(alpha, eval);
      if (beta <= alpha) {
        break;  // Beta cutoff
      }
    }

    // Store the evaluated state in the transposition table
    transposition_table[hash] = best_value;
    return best_value;
  }

  virtual Board::Action generate(Board& b) override {
    int best_action = -1;
    Board::Reward best_value = -std::numeric_limits<Board::Reward>::infinity();
    for (auto& action : b.shuffle_legal_move()) {
      auto b_ = b;
      auto&& [reward, done] = b_.apply(action);
      Board::Reward value =
          reward - principalVariationSearch(
                       b_, max_depth, done,
                       -std::numeric_limits<Board::Reward>::infinity(),
                       std::numeric_limits<Board::Reward>::infinity());
      if (value > best_value) {
        best_value = value;
        best_action = action;
      }
    }
    return best_action;
  };
};

class hybrid_player : player {
 public:
  int time_limit;
  bool mcts_done = false;
  bool negamax_done = false;
  std::unordered_map<Board::Hash, Board::Reward> transposition_table;
  Board::Action nega_generate(Board& b) {
    int best_action = -1;
    Board::Reward best_value = -std::numeric_limits<Board::Reward>::infinity();
    for (auto& action : b.shuffle_legal_move()) {
      auto b_ = b;
      auto&& [reward, done] = b_.apply(action);
      Board::Reward value =
          reward -
          negamaxSearch(b_, done,
                        -std::numeric_limits<Board::Reward>::infinity(),
                        std::numeric_limits<Board::Reward>::infinity());
      if (value > best_value) {
        best_value = value;
        best_action = action;
      }
    }
    return best_action;
  };

  Board::Reward negamaxSearch(const Board& b, bool done, Board::Reward alpha,
                              Board::Reward beta) {
    if (mcts_done) return 0;
    // Check if the state has already been evaluated
    Board::Hash hash = b.hash();
    if (transposition_table.count(hash) > 0) {
      return transposition_table[hash];
    }

    // Check if the search has reached the maximum depth or the game is over
    if (done) {
      return 0;
    }

    Board::Reward best_value = -std::numeric_limits<Board::Reward>::infinity();
    // ... generate possible moves and evaluate them
    for (const Board::Action& action : b.shuffle_legal_move()) {
      Board b_ = b;
      auto&& [r, done] = b_.apply(action);
      Board::Reward eval = r - negamaxSearch(b_, done, -beta, -alpha);
      best_value = std::max(best_value, eval);
      alpha = std::max(alpha, eval);
      if (beta <= alpha) {
        break;  // Beta cutoff
      }
    }
    // Store the evaluated state in the transposition table
    if (!mcts_done) transposition_table[hash] = best_value;
    return best_value;
  };

  hybrid_player(int time_limit = 57) : time_limit(time_limit){};
  virtual Board::Action generate(Board& b) override {
    std::chrono::steady_clock::time_point start =
        std::chrono::steady_clock::now();
    negamax_done = false;
    mcts_done = false;

    Board::Action negamax_result;
    std::thread negamaxThread([&]() {
      negamax_result = nega_generate(b);
      negamax_done = true;
    });

    Board::Action mcts_result;
    std::thread mctsThread([&]() {
      Node* root = new Node(b);
      root->expand();

      while (std::chrono::steady_clock::now() - start <
                 std::chrono::seconds(time_limit) &&
             !negamax_done) {
        Node* selected = root->select();
        selected->expand();
        Board::Reward score = selected->rollout();
        selected->backpropagate(score);
      }

      Board::Reward best_reward =
          -std::numeric_limits<Board::Reward>::infinity();
      for (auto* child : root->children) {
        Board::Reward avg_score = child->value();
        if (avg_score > best_reward) {
          best_reward = avg_score;
          mcts_result = child->action;
        }
      }
      delete root;
      mcts_done = true;
    });

    // Wait for either thread to finish
    if (negamaxThread.joinable()) negamaxThread.join();
    if (mctsThread.joinable()) mctsThread.join();

    // Return the result from the finished thread
    return (negamax_done) ? negamax_result : mcts_result;
  };
};