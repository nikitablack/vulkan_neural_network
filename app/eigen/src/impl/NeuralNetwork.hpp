#pragma once

#include <impl/Layer.hpp>

namespace common {

template <typename T>
class LCG;

}

namespace impl {

class NeuralNetwork {
public:
    NeuralNetwork(std::vector<size_t> const& layerSizes, common::LCG<common::Float>& lcg);

    [[nodiscard]] auto forward(std::vector<common::Float> const& inputValues,  //
                               std::vector<common::Float>& outputValues  //
                               ) noexcept -> bool;

    [[nodiscard]] auto train(std::vector<std::vector<common::Float>> const& input,  //
                             std::vector<uint8_t> const& target,  //
                             size_t epochCount,  //
                             common::Float learningRate  //
                             ) noexcept -> bool;

private:
    [[nodiscard]] auto backward(std::vector<common::Float> const& output,  //
                                std::vector<common::Float> const& expectedOutput,  //
                                common::Float learningRate  //
                                ) noexcept -> bool;

    auto print() const noexcept -> void;

public:
    static auto sigmoid(common::Float v) noexcept -> common::Float;
    static auto sigmoidDerivative(common::Float v) noexcept -> common::Float;

public:
    std::vector<Layer> layers;
};

}  // namespace impl
