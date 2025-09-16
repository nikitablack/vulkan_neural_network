#include <fmt/core.h>

#include <common/LCG.hpp>
#include <impl/Layer.hpp>

namespace impl {

Layer::Layer(size_t neuronCount, size_t inputCount, common::LCG<common::Float>& lcg) noexcept {
    neurons.reserve(neuronCount);

    if (inputCount == 0) {
        neurons.resize(neuronCount);
    } else {
        for (size_t i{0}; i < neuronCount; ++i) {
            neurons.emplace_back(inputCount, lcg);
        }
    }

    delta.resize(neuronCount);
}

[[nodiscard]] auto Layer::activate(Layer const& prevLayer,  //
                                   std::function<auto(common::Float)->common::Float> const& activationFunction  //
                                   ) noexcept -> bool {
    for (auto& currNeuron : neurons) {
        common::Float z{currNeuron.bias};

        for (size_t j{0}; j < prevLayer.neurons.size(); ++j) {
            if (prevLayer.neurons.size() < currNeuron.weights.size()) {
                return false;  // Not enough weights for the neuron
            }

            auto const& prevNeuron{prevLayer.neurons[j]};

            z += currNeuron.weights[j] * prevNeuron.value;
        }

        currNeuron.value = activationFunction(z);
    }

    return true;
}

[[nodiscard]] auto Layer::update(Layer const& prevLayer,  //
                                 common::Float learningRate  //
                                 ) noexcept -> bool {
    for (size_t i{0}; i < neurons.size(); ++i) {
        auto& currNeuron{neurons[i]};

        if (currNeuron.weights.size() != prevLayer.neurons.size()) {
            return false;  // Not enough weights for the neuron
        }

        for (size_t j{0}; j < currNeuron.weights.size(); ++j) {
            currNeuron.weights[j] -= learningRate * delta[i] * prevLayer.neurons[j].value;
        }

        currNeuron.bias -= learningRate * delta[i];
    }

    return true;
}

}  // namespace impl
