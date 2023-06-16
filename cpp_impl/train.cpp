#include <iostream>
#include <ctime>
#include "net.hpp"
#include "episode.hpp"
#include "replay_buffer.hpp"
#include "board.hpp"

int main() {
    std::srand(123);
    // std::srand(std::time(nullptr));
    const size_t EPOCHS = 600;
    Board::Reward lr = 0.01;
    Board::Reward lambda = 0.01;
    ReplayBuffer buffer(6000);
    int n = 30;
    for (int i = 0; i < n; i++) {
        Episode ep;
        buffer.load("trajectory/"+ std::to_string(i) + "_0M1k50000sim.episode");
    }
    buffer.shuffle();
    auto&& [train_set, test_set] = buffer.split(0.8);
    // auto net = TupleNet<3, 8>({{{0,1,2}, {3,4,5}, {6,7,8}, {0,3,6}, {1,4,7}, {2,5,8}, {0,4,8}, {2,4,6}}});
    // auto net = TupleNet<3, 3>({{{0,1,2}, {3,4,5}, {0,4,8}}});
    // auto net = TupleNet<3,16>({{
    //   {0, 1, 2}, {0, 1, 3}, {0, 1, 4}, {0, 1, 5}, {0, 1, 6}, {0, 1, 7},
    //   {0, 1, 8}, {0, 2, 4}, {0, 2, 6}, {0, 2, 7}, {0, 4, 5}, {0, 4, 8},
    //   {0, 5, 7}, {1, 3, 4}, {1, 3, 5}, {1, 4, 7}}});
    auto net = TupleNet<3,8>({{
      {0, 1, 2}, {0, 1, 5}, {0, 1, 6}, 
      {0, 1, 8}, {0, 2, 6}, {0, 4, 5}, {0, 4, 8},
      {0, 5, 7}, }});
    // auto net = TupleNet<4, 6>({{{0,1,2,3}, {0,1,2,4}, {0,1,3,4}, {3,4,5,0}, {3,4,5,1}, {0,4,8,5}}});

    // auto net = TupleNet<3,16>("model/3_16_0.01_0.02.model");
    // Start Training
    for(size_t epoch = 1; epoch <= EPOCHS; ++epoch){
        if(epoch % 10 == 0) {
            lr *= .9;
            lambda *= 1.05;
        }
        Board::Reward train_loss = 0;
        clock_t begin_time = clock();
        train_set.shuffle();
        for(auto ep : train_set){
            Board::Reward target = 0;
            Board::Reward loss = 0;
            const int size = ep.size();
            Board b = ep.init_state;
            for (; ep.history.size(); ep.history.pop_back()) {
                const auto& [action, reward, b_next] = ep.history.back();
                Board::Reward error = target - net.evaluate(b_next);
                // std::cout << target << " " << net.evaluate(b_next) << std::endl;
                loss += std::abs(error);
                net.update_net(b_next, error, lr);
                net.update_weights(b_next, error, lr, lambda);

                target = reward - net.evaluate(b_next);
            }
            net.update_net(b, target - net.evaluate(b), lr);
            net.update_weights(b, target - net.evaluate(b), lr, lambda);
            train_loss += loss / size;
        }
        std::cout.precision(2);
        for(const auto& w : net.weights){
            std::cout << w << " ";
        }std::cout << std::endl;
        for(const auto& w : net.gradient_weights){
            std::cout << w << " ";
        }std::cout << std::endl;
        Board::Reward test_loss = 0;
        
        for(auto ep : test_set){
            Board::Reward target = 0;
            Board::Reward loss = 0;
            const int size = ep.size();
            Board b = ep.init_state;
            for (; ep.history.size(); ep.history.pop_back()) {
                const auto& [action, reward, b_next] = ep.history.back();
                Board::Reward error = target - net.evaluate(b_next);
                loss += std::abs(error);
                target = reward - net.evaluate(b_next);
            }
            test_loss += loss / size;
        }
        std::cout.precision(3);
        std::cout << "Epoch " << epoch << " ( " << Board::Reward(clock() - begin_time) / CLOCKS_PER_SEC << " sec) || train loss: " << train_loss / train_set.size() << " || test loss: " << test_loss / test_set.size() << std::endl;
    }
    // const std::string saved_path = "model/3_16_0.01_0.02.model";
    // net.save(saved_path);
}