#include <cmath>
#include <vector>

class MLP {
 public:
  MLP(const std::vector<int>& layerSizes);
  std::vector<double> forward(const std::vector<double>& input);
  void train(const std::vector<std::vector<double>>& inputs,
             const std::vector<std::vector<double>>& targets, int epochs,
             double learningRate);

 private:
  std::vector<std::vector<double>> weights;
  std::vector<std::vector<double>> biases;
  std::vector<int> layerSizes;
  void backward(const std::vector<double>& input,
                const std::vector<double>& target, double learningRate);
};

MLP::MLP(const std::vector<int>& layerSizes) : layerSizes(layerSizes) {
  // Initialize weights and biases for each layer
  for (int i = 1; i < layerSizes.size(); ++i) {
    int prevLayerSize = layerSizes[i - 1];
    int currentLayerSize = layerSizes[i];

    std::vector<double> layerWeights(prevLayerSize * currentLayerSize);
    std::vector<double> layerBiases(currentLayerSize);

    // Initialize weights and biases (e.g., random initialization)
    // ...

    weights.push_back(layerWeights);
    biases.push_back(layerBiases);
  }
}

std::vector<double> MLP::forward(const std::vector<double>& input) {
  std::vector<double> activations = input;

  // Perform forward propagation through each layer
  for (int i = 0; i < layerSizes.size() - 1; ++i) {
    int inputSize = layerSizes[i];
    int outputSize = layerSizes[i + 1];

    std::vector<double> nextActivations(outputSize);

    // Compute activations for the next layer
    for (int j = 0; j < outputSize; ++j) {
      double activation = biases[i][j];
      for (int k = 0; k < inputSize; ++k) {
        activation += activations[k] * weights[i][k * outputSize + j];
      }
      nextActivations[j] =
          std::max(0.0, activation);  // ReLU activation function
    }

    activations = nextActivations;
  }

  return activations;
}

void MLP::backward(const std::vector<double>& input, const std::vector<double>& target, double learningRate) {
    std::vector<std::vector<double>> layerActivations(layerSizes.size());
    std::vector<std::vector<double>> layerGradients(layerSizes.size());

    // Perform forward pass and store layer activations
    layerActivations[0] = input;
    for (int i = 0; i < layerSizes.size() - 1; ++i) {
        const std::vector<double>& activations = layerActivations[i];
        std::vector<double>& nextActivations = layerActivations[i + 1];

        nextActivations.resize(layerSizes[i + 1]);

        // Compute activations for the next layer
        for (int j = 0; j < layerSizes[i + 1]; ++j) {
            double activation = biases[i][j];
            for (int k = 0; k < layerSizes[i]; ++k) {
                activation += activations[k] * weights[i][k * layerSizes[i + 1] + j];
            }
            nextActivations[j] = std::max(0.0, activation);  // ReLU activation function
        }
    }

    // Compute gradients for the output layer
    const std::vector<double>& outputActivations = layerActivations.back();
    std::vector<double>& outputGradients = layerGradients.back();
    for (int i = 0; i < layerSizes.back(); ++i) {
        outputGradients[i] = outputActivations[i] - target[i];
    }

    // Perform backward pass to compute gradients for hidden layers
    for (int i = layerSizes.size() - 2; i >= 0; --i) {
        const std::vector<double>& activations = layerActivations[i];
        const std::vector<double>& nextGradients = layerGradients[i + 1];
        std::vector<double>& gradients = layerGradients[i];
        std::vector<double>& weightUpdates = weights[i];

        gradients.resize(layerSizes[i]);

        for (int j = 0; j < layerSizes[i]; ++j) {
            double gradient = 0.0;
            for (int k = 0; k < layerSizes[i + 1]; ++k) {
                gradient += nextGradients[k] * weights[i][j * layerSizes[i + 1] + k];
            }
            gradients[j] = (activations[j] > 0.0) ? gradient : 0.0;  // ReLU derivative
        }

        // Update weights
        for (int j = 0; j < layerSizes[i]; ++j) {
            for (int k = 0; k < layerSizes[i + 1]; ++k) {
                weightUpdates[j * layerSizes[i + 1] + k] -= learningRate * activations[j] * nextGradients[k];
            }
        }
    }

    // Apply weight updates
    for (int i = 0; i < weights.size(); ++i) {
        std::vector<double>& layerWeights = weights[i];
        const std::vector<double>& weightUpdates = weightUpdateGradients[i];

        for (int j = 0; j < layerWeights.size(); ++j) {
            layerWeights[j] += weightUpdates[j];
        }
    }
}

void MLP::train(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& targets, int epochs, double learningRate) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (int i = 0; i < inputs.size(); ++i) {
            const std::vector<double>& input = inputs[i];
            const std::vector<double>& target = targets[i];

            backward(input, target, learningRate);
        }
    }
}
