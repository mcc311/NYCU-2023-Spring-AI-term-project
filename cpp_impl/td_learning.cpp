#include <iostream>
#include "net.hpp"
#include "episode.hpp"
#include "replay_buffer.hpp"
#include "board.hpp"
int main() {
    const size_t EPOCHS = 100;
    float lr = 0.01;
    ReplayBuffer buffer(3000);
    int n = 30;
    for (int i = 0; i < n; i++) {
        Episode ep;
        buffer.load("trajectory/"+ std::to_string(i) + "_0M1k50000sim.episode");
    }
    buffer.shuffle();
    auto&& [train_set, test_set] = buffer.split(0.8);
    auto net = NTuple<3, 8>();
    net.set_feats({{{0,1,2}, {3,4,5}, {6,7,8}, {0,3,6}, {1,4,7}, {2,5,8}, {0,4,8}, {2,4,6}}});


    // Start Training
    // for(size_t epoch = 0; epoch < EPOCHS; ++epoch){
    //     float total_loss = 0;
    //     clock_t begin_time = clock();
    //     for(auto ep : train_set){
    //         int target = 0;
    //         float loss = 0;
    //         const int size = ep.size();
    //         Board b = ep.init_state;
    //         for (; ep.history.size(); ep.history.pop_back()) {
    //             const auto& [action, reward, b_next] = ep.history.back();
    //             float error = target - net.evaluate(b_next);
    //             loss += error;
    //             net.update(b_next, error, lr);
    //             target = reward - net.evaluate(b_next);
    //         }
    //         net.update(b, target - net.evaluate(b), lr);
    //         total_loss += loss / size;
    //     }
    //     std::cout << "Epoch " << epoch << " ( " << float(clock() - begin_time) / CLOCKS_PER_SEC << " sec) || loss: " << total_loss / train_set.size() << std::endl;
    // }
}