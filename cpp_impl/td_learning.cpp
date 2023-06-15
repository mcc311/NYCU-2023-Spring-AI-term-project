#include <iostream>
#include "net.hpp"
#include "episode.hpp"
int main() {
    ReplayBuffer buffer(1000);
    int n = 30;
    for (int i = 0; i < n; i++) {
        Episode ep;
        buffer.load("trajectory/"+ std::to_string(i) + "_0M1k50000sim.episode");
    }
    std::cout << buffer.size() << std::endl;
    std::cout << buffer.sample();
}