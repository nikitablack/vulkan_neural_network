#include <impl/Layer.hpp>
#include <random>

namespace impl {

Layer::Layer(size_t neuronCount, size_t inputCountArg) noexcept : inputCount{inputCountArg} {
    neurons.reserve(neuronCount);

    if (inputCountArg == 0) {
        neurons.resize(neuronCount);
    } else {
        std::random_device rd{};
        std::mt19937 gen{rd()};
        std::uniform_real_distribution<double> dist{-1.0, 1.0};

        for (size_t i{0}; i < neuronCount; ++i) {
            double const value{dist(gen)};
            double const bias{dist(gen)};
            neurons.emplace_back(value, bias, inputCountArg);
        }
    }
}

[[nodiscard]] auto Layer::activate(Layer const& prevLayer,  //
                                   std::function<auto(double)->double> const& activationFunction  //
                                   ) noexcept -> bool {
    for (auto& currNeuron : neurons) {
        double sum{currNeuron.bias};

        for (size_t j{0}; j < prevLayer.neurons.size(); ++j) {
            if (prevLayer.neurons.size() < currNeuron.weights.size()) {
                return false;  // Not enough weights for the neuron
            }

            auto const& prevNeuron{prevLayer.neurons[j]};

            sum += currNeuron.weights[j] * prevNeuron.value;
        }

        currNeuron.value = activationFunction(sum);
    }

    return true;
}

[[nodiscard]] auto Layer::update(Layer const& prevLayer,  //
                                 double learningRate,  //
                                 std::vector<double> const& delta  //
                                 ) noexcept -> bool {
    if (delta.size() != neurons.size()) {
        return false;  // Mismatch in delta size
    }

    for (size_t i{0}; i < neurons.size(); ++i) {
        auto& currNeuron{neurons[i]};

        if (currNeuron.weights.size() != prevLayer.neurons.size()) {
            return false;  // Not enough weights for the neuron
        }

        for (size_t j{0}; j < currNeuron.weights.size(); ++j) {
            currNeuron.weights[j] -= learningRate * delta[i] * prevLayer.neurons[j].value;
            currNeuron.bias -= learningRate * delta[i];
        }
    }

    return true;
}

}  // namespace impl
