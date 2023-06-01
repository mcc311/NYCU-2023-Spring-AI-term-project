#include "agent.hpp"
float player::evaluate(const board& b) { return 0; };
int player::generate(board& b) { return 0; };
void player::update(const board& b, const float error,
                    const float alpha = 0.001) {
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
player& player::operator=(const player& who) {
  for (int i = 0; i < 8; i++) {
    std::copy(who.net[i].begin(), who.net[i].end(), net[i].begin());
  }
  return *this;
}

void player::save(const std::string& path) {
  for (int i = 0; i < feat_num; i++) {
    std::ofstream out;
    out.open((path + std::to_string(i) + ".bin").c_str(),
             std::ios::out | std::ios::binary | std::ios::trunc);
    if (out.is_open()) {
      out.write((char*)(net[i].data()), net[i].size());
      out.flush();
      out.close();
    }
  }
};
void player::load(const std::string& path) {
  for (int i = 0; i < feat_num; i++) {
    std::ifstream in;
    in.open((path + std::to_string(i) + ".bin").c_str(),
            std::ios::in | std::ios::binary);
    if (in.is_open()) {
      in.read((char*)(net[i].data()), net[i].size());
      in.close();
    }
  }
};

  td_player::td_player(){};
  float td_player::evaluate(const board& b) {
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
  int td_player::generate(board& b) {
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
  td_player& td_player::operator=(const td_player& who) {
    for (int i = 0; i < 8; i++) {
      std::copy(who.net[i].begin(), who.net[i].end(), net[i].begin());
    }
    return *this;
  }

  void td_player::save(const std::string& path) {
    for (int i = 0; i < feat_num; i++) {
      std::ofstream out;
      out.open((path + std::to_string(i) + ".bin").c_str(),
               std::ios::out | std::ios::binary | std::ios::trunc);
      if (out.is_open()) {
        out.write((char*)(net[i].data()), net[i].size());
        out.flush();
        out.close();
      }
    }
  };
  void td_player::load(const std::string& path) {
    for (int i = 0; i < feat_num; i++) {
      std::ifstream in;
      in.open((path + std::to_string(i) + ".bin").c_str(),
              std::ios::in | std::ios::binary);
      if (in.is_open()) {
        in.read((char*)(net[i].data()), net[i].size());
        in.close();
      }
    }
  };
