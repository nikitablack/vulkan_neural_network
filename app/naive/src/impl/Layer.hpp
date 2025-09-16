#pragma once

#include <functional>
#include <impl/Neuron.hpp>
#include <vector>

namespace common {

template <typename T>
class LCG;

}

namespace impl {

class Layer {
public:
    Layer(size_t neuronCount, size_t inputCountArg, common::LCG<common::Float>& lcg) noexcept;

public:
    [[nodiscard]] auto activate(Layer const& prevLayer,  //
                                std::function<auto(common::Float)->common::Float> const& activationFunction  //
                                ) noexcept -> bool;

    [[nodiscard]] auto update(Layer const& prevLayer,  //
                              common::Float learningRate  //
                              ) noexcept -> bool;

public:
    std::vector<Neuron> neurons{};
    std::vector<common::Float> delta{};
};

}  // namespace impl
