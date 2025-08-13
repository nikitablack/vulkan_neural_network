#include <impl/Neuron.hpp>
#include <random>

namespace impl {

Neuron::Neuron(Float valueArg, Float biasArg, size_t inputCountArg) noexcept
    : value{valueArg}, bias{biasArg}, weights(inputCountArg) {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::uniform_real_distribution<Float> dist{-1.0, 1.0};

    for (size_t i{0}; i < weights.size(); ++i) {
        Float const w{dist(gen)};
        weights[i] = w;
    }
}

}  // namespace impl
