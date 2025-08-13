#pragma once

#include <functional>
#include <impl/Neuron.hpp>
#include <vector>

namespace impl {

class Layer {
public:
    Layer() = default;

    Layer(size_t neuronCount, size_t inputCountArg) noexcept;

public:
    [[nodiscard]] auto activate(Layer const& prevLayer,  //
                                std::function<auto(Float)->Float> const& activationFunction  //
                                ) noexcept -> bool;

    [[nodiscard]] auto update(Layer const& prevLayer,  //
                              Float learningRate,  //
                              std::vector<Float> const& delta  //
                              ) noexcept -> bool;

public:
    std::vector<Neuron> neurons{};
    size_t inputCount{0};
};

}  // namespace impl
