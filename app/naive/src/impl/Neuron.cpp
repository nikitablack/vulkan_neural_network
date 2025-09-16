#include <common/LCG.hpp>
#include <impl/Neuron.hpp>

namespace impl {

Neuron::Neuron(size_t inputCount, common::LCG<common::Float>& lcg) noexcept : weights(inputCount) {
    for (size_t i{0}; i < weights.size(); ++i) {
        common::Float const w{lcg.next()};
        weights[i] = w;
    }

    bias = lcg.next();
}

}  // namespace impl
