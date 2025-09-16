#pragma once

#include <Eigen/Dense>
#include <common/Float.hpp>
#include <functional>
#include <vector>

namespace common {

template <typename T>
class LCG;

}

namespace impl {

class Layer {
public:
    using MatrixX = Eigen::Matrix<common::Float, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorX = Eigen::Matrix<common::Float, Eigen::Dynamic, 1>;

public:
    Layer() noexcept = default;

    Layer(size_t neuronCount, size_t inputCount, common::LCG<common::Float>& lcg) noexcept;

public:
    [[nodiscard]] auto activate(Layer const& prevLayer,  //
                                std::function<auto(common::Float)->common::Float> const& activationFunction  //
                                ) noexcept -> bool;

    [[nodiscard]] auto update(Layer const& prevLayer,  //
                              common::Float learningRate  //
                              ) noexcept -> bool;

    size_t size() const noexcept;

public:
    MatrixX weights{};
    MatrixX biases{};
    MatrixX values{};
    MatrixX delta{};
};

}  // namespace impl
