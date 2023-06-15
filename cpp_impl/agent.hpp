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
  std::unordered_map<Board::Hash, Board::Reward> transpositionTable;
  nega_player(int max_depth = 3, bool heuristic = false)
      : max_depth(max_depth), heuristic(heuristic){};

  Board::Reward evaluate(const Board& b) {
    // return net.evaluate(b);
    return mcts_estimate(b, 100);
  }
  Board::Reward negamaxSearch(const Board& b, int depth, bool done,
                              Board::Reward alpha, Board::Reward beta) {
    // Check if the state has already been evaluated
    Board::Hash hashValue = b.hash();
    if (transpositionTable.count(hashValue) > 0) {
      return transpositionTable[hashValue];
    }

    // Check if the search has reached the maximum depth or the game is over
    if (done) {
      return 0;
    }
    // if (depth == 0) {
    //   transpositionTable[hashValue] = evaluate(b);
    //   return transpositionTable[hashValue];
    // }

    Board::Reward maxEval = -std::numeric_limits<Board::Reward>::infinity();
    // ... generate possible moves and evaluate them
    for (const Board::Action& action : b.shuffle_legal_move(heuristic)) {
      Board b_ = b;
      auto&& [r, done] = b_.apply(action);
      Board::Reward eval =
          r - negamaxSearch(b_, depth - 1, done, -beta, -alpha);
      maxEval = std::max(maxEval, eval);
      alpha = std::max(alpha, eval);
      if (beta <= alpha) {
        break;  // Beta cutoff
      }
    }
    // Store the evaluated state in the transposition table
    transpositionTable[hashValue] = maxEval;
    return maxEval;
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
    Board::Hash hashValue = b.hash();
    if (transpositionTable.count(hashValue) > 0) {
      return transpositionTable[hashValue];
    }

    // Check if the search has reached the maximum depth or the game is over
    if (done) {
      return 0;
    }
    if (depth == 0) {
      return evaluate(b);
    }

    Board::Reward maxEval = -std::numeric_limits<Board::Reward>::infinity();
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

      maxEval = std::max(maxEval, eval);
      alpha = std::max(alpha, eval);
      if (beta <= alpha) {
        break;  // Beta cutoff
      }
    }

    // Store the evaluated state in the transposition table
    transpositionTable[hashValue] = maxEval;
    return maxEval;
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
#define c (std::cout << __LINE__ << std::endl)
class hybrid_player : nega_player { 
 public:
  int time_limit;
  hybrid_player(int time_limit = 59) : nega_player(), time_limit(time_limit){};
  virtual Board::Action generate(Board& b) override {
c;    Board::Action negamax_result;
c;    bool negamax_done = false;
c;    std::thread negamaxThread([&]() {
c;      negamax_result = nega_player::generate(b);
c;      negamax_done = true;
c;    });
c;
c;    Board::Action mcts_result;
c;    std::thread mctsThread([&]() {
c;      Node* root = new Node(b);
c;      root->expand();
c;
c;      std::chrono::steady_clock::time_point start =
          std::chrono::steady_clock::now();
c;      while (std::chrono::steady_clock::now() - start <
            std::chrono::seconds(time_limit) && !negamax_done) {
c;        Node* selected = root->select();
c;        selected->expand();
c;        Board::Reward score = selected->rollout();
c;        selected->backpropagate(score);
c;      }
c;
c;      Board::Reward best_reward =
          -std::numeric_limits<Board::Reward>::infinity();
c;      for (auto* child : root->children) {
c;        Board::Reward avg_score = child->value();
c;        if (avg_score > best_reward) {
c;          best_reward = avg_score;
c;          mcts_result = child->action;
c;        }
c;      }
        delete root;
c;    });
c;
c;    // Wait for either thread to finish
c;    if (negamaxThread.joinable()) negamaxThread.join();
c;    if (mctsThread.joinable()) mctsThread.join();
c;
c;    // Return the result from the finished thread
c;    return (negamax_done) ? negamax_result : mcts_result;
  }
};
