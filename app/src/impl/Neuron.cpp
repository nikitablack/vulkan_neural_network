#include <impl/Neuron.hpp>
#include <random>

namespace impl {

Neuron::Neuron(size_t inputCount) noexcept : weights(inputCount) {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::uniform_real_distribution<Float> dist{-1.0, 1.0};

    for (size_t i{0}; i < weights.size(); ++i) {
        Float const w{dist(gen)};
        weights[i] = w;
    }

    bias = dist(gen);
}

}  // namespace impl
