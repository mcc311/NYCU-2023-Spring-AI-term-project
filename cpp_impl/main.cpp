#include <algorithm>
#include <array>
#include <execution>
#include <fstream>
#include <iostream>
#include <random>
#include <ranges>
#include <vector>

class board {
 public:
  using reward = int;
  using action = int;
  uint64_t raw = 0;
  inline int get(const int index, const int isomorphic = 0) const noexcept {
    static constexpr int isom_table[8][9] = {
        {0, 1, 2, 3, 4, 5, 6, 7, 8}, {2, 5, 8, 1, 4, 7, 0, 3, 6},
        {8, 7, 6, 5, 4, 3, 2, 1, 0}, {6, 3, 0, 7, 4, 1, 8, 5, 2},
        {2, 1, 0, 5, 4, 3, 8, 7, 6}, {0, 3, 6, 1, 4, 7, 2, 5, 8},
        {6, 7, 8, 3, 4, 5, 0, 1, 2}, {8, 5, 2, 7, 4, 1, 6, 3, 0}};
    return int((raw >> (isom_table[isomorphic][index] * 7)) & 0b1111111ull);
  }
  inline void set(const uint8_t index, uint64_t value) {
    uint64_t mask = ~(0b1111111ull << (index * 7));
    raw = (raw & mask) | (value << (index * 7));
  }
  std::tuple<const int (&)[3], const int> action2idx(int action) {
    static constexpr int idxs[6][3] = {{0, 1, 2}, {3, 4, 5}, {6, 7, 8},
                                       {0, 3, 6}, {1, 4, 7}, {2, 5, 8}};
    return {idxs[action % 6], /*minus= */ action / 6 + 1};
  }
  bool legal(int action) {
    auto&& [idxs, minus] = action2idx(action);
    bool legal = true;
    for (auto& idx : idxs) {
      legal &= get(idx) >= minus;
    }
    return legal;
  };

  std::tuple<reward, bool> terminated() {
    static constexpr uint64_t bonus_pattern[8] = {
        /*
        0, x, x
        0, x, x
        0, x, x
        */
        0b0111111100000000000000111111100000000000000111111100000000000000ull,
        /*
        x, 0, x
        x, 0, x
        x, 0, x
        */
        0b0000000011111110000000000000011111110000000000000011111110000000ull,
        /*
        x, x, 0
        x, x, 0
        x, x, 0
        */
        0b0000000000000001111111000000000000001111111000000000000001111111ull,
        /*
        0, 0, 0
        x, x, x
        x, x, x
        */
        0b0111111111111111111111000000000000000000000000000000000000000000ull,
        /*
        x, x, x
        0, 0, 0
        x, x, x
        */
        0b0000000000000000000000111111111111111111111000000000000000000000ull,
        /*
        x, x, x
        x, x, x
        0, 0, 0
        */
        0b0000000000000000000000000000000000000000000111111111111111111111ull,
        /*
        0, x, x
        x, 0, x
        x, x, 0
        */
        0b0111111100000000000000000000011111110000000000000000000001111111ull,
        /*
        x, x, 0
        x, 0, x
        0, x, x
        */
        0b0000000000000001111111000000011111110000000111111100000000000000ull,
    };
    static constexpr reward bonus = 15;
    static constexpr reward penalty = -7;

    // check if it's terminated with bonus
    bool is_bonus = false;
    for (auto& pattern : bonus_pattern) {
      is_bonus |= !(raw & pattern);
    }
    // check if it's terminated with penalty
    bool is_penalty = false;
    int count = 0;
    for (int i = 0; i < 6; i++) {
      is_penalty |= legal(i);
    }
    is_penalty = !is_penalty;
    return {bonus * is_bonus + penalty * (is_penalty & !is_bonus),
            is_bonus || is_penalty};
  }

  std::tuple<reward, bool> apply(int action) {
    auto&& [idxs, minus] = action2idx(action);
    for (auto& idx : idxs) {
      set(idx, get(idx) - minus);
    }
    auto&& [rt, done] = terminated();
    return {rt - minus, done};
  }
  board& operator=(const board& b) {
    raw = b.raw;
    return *this;
  }
};

std::ostream& operator<<(std::ostream& os, const board& b) {
  for (int i = 0; i < 9; i++) {
    os << int(b.get(i)) << ((i + 1) % 3 ? "\t" : "\n");
  }
  return os;
}

std::vector<int> shuffle_actions() {
  int N = 18;
  std::vector<int> numbers(N);
  std::iota(numbers.begin(), numbers.end(), 0);

  // Shuffle the numbers randomly
  std::random_device rd;
  std::mt19937 gen(rd());
  std::ranges::shuffle(numbers, gen);
  return numbers;
}

class player {
 private:
  static constexpr int feat_num = 8;
  static constexpr int isom_num = 8;
  static constexpr int feat_idxs[feat_num][3] = {
      {0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {0, 3, 6},
      {1, 4, 7}, {2, 5, 8}, {0, 4, 8}, {2, 4, 6}};
  static constexpr int net_size = 100 * 100 * 100;

 public:
  player() : net(feat_num, std::vector<float>(net_size, 0)){};
  std::vector<std::vector<float> > net;
  float evaluate(const board& b) {
    float value = 0;
    // static constexpr auto feat_range = std::ranges::views::iota(0, feat_num);
    // std::mutex m;
    // std::for_each(std::execution::par, feat_range.begin(), feat_range.end(),
    //               [&value, &b, this, &m](const int& i) {
    //                 for (int isom = 0; isom < isom_num; isom++) {
    //                   uint32_t feat = 0;
    //                   for (const auto& f : feat_idxs[i]) {
    //                     feat = feat * 100;
    //                     feat += b.get(f, isom);
    //                   }
    //                   std::lock_guard<std::mutex> guard(m);
    //                   value += net[i][feat];
    //                 }
    //               });

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
  int generate(board& b) {
    int best_action = -1;
    float best_value = -MAXFLOAT;
    for (auto& action :
         shuffle_actions() |
             std::views::filter([&b](int action) { return b.legal(action); })) {
      auto b_ = b;
      // if (!b_.legal(action)) continue;
      auto&& [reward, done] = b_.apply(action);
      auto value = reward - evaluate(b_);  // for two player
      if (value > best_value) {
        best_value = value;
        best_action = action;
      }
    }
    return best_action;
  }
  void update(const board& b, float error, float alpha = 0.01) {
    // static constexpr auto feat_range = std::ranges::views::iota(0, feat_num*isom_num);
    // std::for_each(std::execution::par, feat_range.begin(), feat_range.end(),
    //               [&alpha, &error, &b, this](const int& i) {
    //                   uint32_t feat = 0;
    //                 for (const auto& f : feat_idxs[i%8]) {
    //                   feat = feat * 100;
    //                   feat += b.get(f, i/8);
    //                 }
    //                 net[i%8][feat] += alpha * error / (feat_num * isom_num);
    //               });
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

  }
  player& operator=(const player& who) {
    for (int i = 0; i < 8; i++) {
      std::copy(who.net[i].begin(), who.net[i].end(), net[i].begin());
    }
    return *this;
  }

  void save(const std::string& path) {
    for (int i = 0; i < feat_num; i++) {
      std::ofstream out;
      out.open((path + std::to_string(i) + ".bin").c_str(),
               std::ios::out | std::ios::binary | std::ios::trunc);

      if (out.is_open()) {
        //   std::ostream_iterator<float> output_iterator(out, " ");
        //   std::copy(net[i].begin(), net[i].end(), output_iterator);
        // }
        out.write((char*)(net[i].data()), net[i].size());
        out.flush();
        out.close();
      }
    }
  };
  void load(const std::string& path) {
    for (int i = 0; i < feat_num; i++) {
      std::ifstream in;
      in.open((path + std::to_string(i) + ".bin").c_str(),
              std::ios::in | std::ios::binary);
      if (in.is_open()) {
        // std::istream_iterator<float> input_iterator(in), end;
        // std::copy(input_iterator , end, net[i].begin());
        in.read((char*)(net[i].data()), net[i].size());
        in.close();
      }
    }
  };
};

class episode {
 public:
  using step = std::tuple<board::action, board::reward,
                          board>;  // St, At, R_t, St+1
  std::vector<step> history;
  float scores[2] = {};
  void add(const step& s) { history.push_back(s); }
  int win() {
    if (scores[0] > scores[1]) return 0;
    if (scores[0] < scores[1]) return 1;
    return -1;
  }
};

int evaluate(const board& b) { return 0; };

int alphaBetaMin(board, board::reward, board::reward, int);
int alphaBetaMax(board, board::reward, board::reward, int);

static int node_count = 0;

board::reward alphaBetaMax(board b, board::reward alpha, board::reward beta,
                           int depthleft) {
  if (depthleft == 0)
    return evaluate(b);  // Reach end depth or terminal condition.
  for (const auto& action : shuffle_actions()) {
    if (!b.legal(action)) continue;
    board b_ = b;
    node_count += 1;
    auto&& [r, done] = b_.apply(action);
    int score = r + alphaBetaMin(b_, alpha, beta, done ? 0 : depthleft - 1);
    if (score >= beta) return beta;    // fail hard beta-cutoff
    if (score > alpha) alpha = score;  // alpha acts like max in MiniMax
  }
  return alpha;
}

board::reward alphaBetaMin(board b, board::reward alpha, board::reward beta,
                           int depthleft) {
  if (depthleft == 0) return -evaluate(b);
  for (const auto& action : shuffle_actions()) {
    if (!b.legal(action)) continue;
    board b_ = b;
    node_count += 1;
    auto&& [r, done] = b_.apply(action);
    int score = r + alphaBetaMax(b_, alpha, beta, done ? 0 : depthleft - 1);
    if (score <= alpha) return alpha;  // fail hard alpha-cutoff
    if (score < beta) beta = score;    // beta acts like min in MiniMax
  }
  return beta;
}

episode play_an_episode(player (&players)[2], const int play_first = 0,
                        const bool verbose = false) {
  board b;
  for (int i = 0; i < 9; i++) {
    b.set(i, std::rand() % 50 + 50);
  };

  episode ep;
  int pid = play_first;
  while (true) {
    int action = players[pid].generate(b);
    if (action == -1) std::cout << b;
    auto&& [reward, done] = b.apply(action);
    // if (verbose) {
    //   std::cout << b_prev;
    //   std::cout << "Select: " << action % 6 << " Substract: " << action / 6 +
    //   1
    //             << " Reward: " << reward << "\n";
    // }
    ep.add({action, reward, b});
    ep.scores[pid] += reward;
    if (done) break;
    pid = (pid + 1) % 2;
  }
  return ep;
}

void test_player0(player (&players)[2], int num_to_play = 100) {
  int win = 0;
  for (int i_episode = 0; i_episode < num_to_play / 2; i_episode++) {
    episode ep = play_an_episode(players, 0);
    if (ep.win() == 0) win++;
  }
  for (int i_episode = 0; i_episode < num_to_play / 2; i_episode++) {
    episode ep = play_an_episode(players, 1);
    if (ep.win() == 0) win++;
  }
  std::cout << "Win rate: " << float(win) / num_to_play << "\n";
}

void test() {
  board init_b;
  for (int i = 0; i < 9; i++) {
    init_b.set(i, 10);
  }
  std::cout << init_b;
  for (int action = 0; action < 18; action++) {
    board b;
    for (int i = 0; i < 9; i++) {
      b.set(i, 10);
    }
    b.apply(action);
    std::cout << "Select: " << action % 6 << " Substract: " << action / 6 + 1
              << "\n";
    std::cout << b;
  }
  board b = init_b;
  std::cout << "Operator= overloading test: \n";
  std::cout << "Init:\n" << init_b << "Assigned to:\n" << b;
  for (int action = 0; action < 18; action++) {
    board b;
    b = init_b;
    b.apply(action);
    std::cout << "Select: " << action % 6 << " Substract: " << action / 6 + 1
              << "\n";
    std::cout << b;
  }

  std::cout << "\nTermination Test:\n";
  int test_pattern[9][3] = {{0, 1, 2}, {3, 4, 5}, {6, 7, 8},
                            {0, 3, 6}, {1, 4, 7}, {2, 5, 8},
                            {0, 4, 8}, {2, 4, 6}, {1, 5, 6}};
  for (auto& pattern : test_pattern) {
    b = init_b;
    b.set(pattern[0], 0);
    b.set(pattern[1], 0);
    b.set(pattern[2], 0);
    std::cout << b;
    auto&& [r, t] = b.terminated();
    if (t)
      std::cout << "Terminated with reward: " << r << std::endl;
    else
      std::cout << "Not terminated. Reward: " << r << std::endl;
  }
  b = init_b;
  const clock_t begin_time = clock();
  alphaBetaMax(b, -1000000, 100000, 7);
  float elapse = float(clock() - begin_time) / CLOCKS_PER_SEC;

  std::cout << "Takes " << elapse << " secs. And explores " << node_count
            << " nodes. " << float(node_count) / elapse << " node/sec\n";
};

int main() {
  // static std::vector<episode> replay_buffer;
  // replay_buffer.reserve(1000);
  player players[2];
  // players[0].load("test");
  clock_t begin_time = clock();
  for (int i_episode = 0; i_episode < 2000000; i_episode++) {
    if (i_episode % 10000 == 0) {
      float elapse = float(clock() - begin_time) / CLOCKS_PER_SEC;
      player test_p[2];
      test_p[0] = players[0];
      std::cout << "Episode " << i_episode << " ( " << elapse << " sec) ";
      test_player0(test_p, 100);
      begin_time = clock();
    }
    episode ep = play_an_episode(players);
    float target = 0;
    auto& who = players[0];
    for (; ep.history.size(); ep.history.pop_back()) {
      const auto& [action, reward, b_next] = ep.history.back();
      float error = target - who.evaluate(b_next);
      who.update(b_next, error);
      target = reward - who.evaluate(b_next);
    }
    players[1] = players[0];
  }
  players[0].save("test");
}