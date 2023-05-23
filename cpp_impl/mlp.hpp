#include <numeric>
#include <vector>

class layer {
  inline static auto sigmoid(float x) { return 1.0f / (1 + exp(-x)); };
  inline static auto d_sigmoid(float x) { return (x * (1 - x)); };
  using neuron = std::vector<float>;
  layer(int input_dim, int output_dim)
      : input_dim(input_dim),
        output_dim(output_dim),
        weights(output_dim, std::vector<float>(input_dim, 0)){};
  std::vector<std::vector<float> > weights;
  std::vector<float> bias;
  std::vector<std::vector<float> > delta_w;
  int input_dim;
  int output_dim;

  std::vector<float> forward(std::vector<float> input) {
    std::vector<float> output = std::vector<float>(output_dim, 0);
    for (int i = 0; i < output_dim; i++) {
      output[i] =
          std::inner_product(input.begin(), input.end(), weights[i].begin(), 0);
    };
    std::transform(output.begin(), output.end(), bias.begin(), output.begin(),
                   std::plus<float>());
    return output;
  };
  void backward(std::vector<float> output, std::vector<float> truth) {}
};
