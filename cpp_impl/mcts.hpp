#pragma once
#include <math.h>

#include <algorithm>
#include <numeric>
#include <ranges>
#include <vector>

#include "board.hpp"
#include "utils.hpp"

class Node {
 public:
  Board state;   // St
  float reward;  // Rt
  bool terminated;
  Board::Action action;

  float total_score;  // Rt+1 - Rt+2 + Rt+3 - Rt+4 +..., for all simulations
  int visits;
  std::vector<Node*> children;
  Node* parent;
  // Constructor
  Node(const Board& state, const Board::Action action = -1,
       const float reward = 0.0f, const bool done = false,
       Node* parent = nullptr)
      : state(state),
        reward(reward),
        terminated(done),
        action(action),
        total_score(0.0f),
        visits(0),
        parent(parent){};

  void expand() {
    for (const auto& action : this->state.shuffle_legal_move()) {
      Board next_state = this->state;
      auto&& [r, done] = next_state.apply(action);
      Node* child = new Node(next_state, action, r, done, this);
      this->children.push_back(child);
    }
  };
  float value() { return reward - total_score / (visits + 1e-5); }
  Node* select() {
    Node* current = this;

    while (!current->terminated && !current->children.empty()) {
      float best_ucb1 = -std::numeric_limits<float>::infinity();
      Node* best_child = nullptr;

      auto [min_value, max_value] = std::ranges::minmax(
          std::views::transform(current->children, &Node::value));
      // std::cout << min_value << ' ' << max_value << '\n';

      for (auto* child : current->children) {
        if (child->visits == 0 || current->children.size() == 1) {
          best_child = child;
          break;
        }
        float value = (child->value() - min_value + 1e-3) /
                      (max_value - min_value + 1e-3);
        float ucb1 =
            value + 1.25 * std::sqrt(2.0f * std::log(current->visits + 1e-5) /
                                     (child->visits + 1e-5));
        if (ucb1 > best_ucb1) {
          best_ucb1 = ucb1;
          best_child = child;
        }
      }
      current = best_child;
    }

    return current;
  };
  float rollout() {
    Board current = this->state;
    if (this->terminated) return 0.0f;
    bool done = false;
    int who = 1;
    float score = 0;

    while (!done) {
      for (const auto& action : current.shuffle_legal_move()) {
        auto&& [r, d] = current.apply(action);
        done = d;
        score += r * who;
        who *= -1;
        break;
      }
    }
    return score;
  };
  void backpropagate(float score) {
    Node* current = this;
    while (current != nullptr) {
      current->visits++;
      current->total_score += score;
      score = current->reward - score;
      current = current->parent;
    }
  };

  ~Node() { for (auto* child : children) delete child; }
};

Board::Action monte_carlo_tree_search(const Board& state, int sim_count) {
  Node* root = new Node(state);
  root->expand();

  for (int i = 0; i < sim_count; i++) {
    Node* selected = root->select();
    selected->expand();
    float score = selected->rollout();
    selected->backpropagate(score);
  }

  // Select the best action based on the statistics of the root Node's children
  float best_reward = -std::numeric_limits<float>::infinity();
  Board::Action best_action = -1;

  for (auto* child : root->children) {
    float avg_score =
        child->reward - child->total_score / (child->visits + 1e-5);
    // float avg_score = child->visits + 1e-5;
    if (avg_score > best_reward) {
      best_reward = avg_score;
      best_action = child->action;
    }
  }
  delete root;
  return best_action;
}