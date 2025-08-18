#include <impl/Layer.hpp>

namespace impl {

Layer::Layer(size_t neuronCount, size_t inputCount) noexcept {
    neurons.reserve(neuronCount);

    if (inputCount == 0) {
        neurons.resize(neuronCount);
    } else {
        for (size_t i{0}; i < neuronCount; ++i) {
            neurons.emplace_back(inputCount);
        }
    }
}

[[nodiscard]] auto Layer::activate(Layer const& prevLayer,  //
                                   std::function<auto(Float)->Float> const& activationFunction  //
                                   ) noexcept -> bool {
    for (auto& currNeuron : neurons) {
        Float z{currNeuron.bias};

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
                                 Float learningRate,  //
                                 std::vector<Float> const& delta  //
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
        }

        currNeuron.bias -= learningRate * delta[i];
    }

    return true;
}

}  // namespace impl
