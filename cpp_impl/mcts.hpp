#pragma once
#include <math.h>

#include <numeric>
#include <vector>

#include "board.hpp"
#include "utils.hpp"

class node {
 public:
  board state;   // St
  float reward;  // Rt
  bool terminated;
  board::action action;

  float total_score;  // Rt+1 - Rt+2 + Rt+3 - Rt+4 +..., for all simulations
  int visits;
  std::vector<node*> children;
  node* parent;
  // Constructor
  node(const board& state, const board::action action = -1,
       const float reward = 0.0f, const bool done = false,
       node* parent = nullptr)
      : state(state),
        action(action),
        reward(reward),
        total_score(0.0f),
        visits(0),
        terminated(done),
        parent(parent){};

  void expand() {
    for (const auto action : this->state.shuffle_legal_move()){
      board next_state = this->state;
      auto&& [r, done] = next_state.apply(action);
      node* child = new node(next_state, action, r, done, this);
      this->children.push_back(child);
    }
  };
  node* select() {
    node* current = this;
    while (!current->children.empty()) {
      // Select child node based on UCB1 algorithm or another selection policy
      float best_ucb1 = -std::numeric_limits<float>::infinity();
      node* best_child = nullptr;
      for (auto* child : current->children) {
        if (child->visits == 0) {
          best_child = child;
          break;
        }
        float ucb1 = child->reward +
                     child->total_score / (child->visits + 1e-5) +
                     std::sqrt(2.0f * std::log(current->visits + 1e-5) /
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
  void backpropagate(float score) {
    node* current = this;

    while (current != nullptr) {
      current->visits++;
      current->total_score += score;
      score = current->reward - score;
      current = current->parent;
    }
  };
  float rollout() {
    board current = this->state;
    bool done = false;
    int who = 1;
    float score = 0;

    while (!done) {
      // Generate a random legal action
      for (const auto& action : current.shuffle_legal_move()){
        auto&& [r, d] = current.apply(action);
        done = d;
        score += r * who;
        who *= -1;
        break;
      }
    }

    return score;
  };
  void clear() {
    for (auto* child : children) {
      child->clear();
    }
    delete this;
  }
};

board::action monte_carlo_tree_search(const board& state, int sim_count) {
  node* root = new node(state);
  root->expand();

  for (int i = 0; i < sim_count; i++) {
    node* selected = root->select();
    selected->expand();
    float score = selected->rollout();
    selected->backpropagate(score);
  }

  // Select the best action based on the statistics of the root node's children
  float best_reward = -std::numeric_limits<float>::infinity();
  board::action best_action = -1;

  for (auto* child : root->children) {
    float avg_score =
        child->reward - child->total_score / (child->visits + 1e-5);

    if (avg_score > best_reward) {
      best_reward = avg_score;
      best_action = child->action;
    }
  }
  // Clean up memory
  root->clear();
  return best_action;
}