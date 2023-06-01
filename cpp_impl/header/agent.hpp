#pragma once
#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <random>
#include <ranges>
#include <vector>
#include "board.hpp"
#include "utils.h"

class player{
protected:
  static constexpr int feat_num = 8;
  static constexpr int isom_num = 8;
  static constexpr int feat_idxs[feat_num][3] = {
      {0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {0, 3, 6},
      {1, 4, 7}, {2, 5, 8}, {0, 4, 8}, {2, 4, 6}};
  static constexpr int net_size = 100 * 100 * 100;
  std::vector<std::vector<float> > net;
public:
  player() : net(feat_num, std::vector<float>(net_size, 0)){};
  virtual float evaluate(const board& b){return 0;};
  virtual int generate(board& b){return 0;};
  void update(const board& b, const float error, const float alpha = 0.001);
    player& operator=(const player& who);

  void save(const std::string& path);
  void load(const std::string& path) ;
};
class td_player: public player{
 public:
  td_player(){};
  virtual float evaluate(const board& b);
  virtual int generate(board& b);
  td_player& operator=(const td_player& who);

  void save(const std::string& path);
  void load(const std::string& path) ;
};
