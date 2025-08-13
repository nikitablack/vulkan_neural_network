#pragma once

#include <Eigen/Dense>
#include <functional>
#include <impl/Float.hpp>
#include <vector>

namespace impl {

class Layer {
public:
    using MatrixX = Eigen::Matrix<Float, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorX = Eigen::Matrix<Float, Eigen::Dynamic, 1>;

public:
    Layer() noexcept = default;

    Layer(size_t neuronCount, size_t inputCount) noexcept;

public:
    [[nodiscard]] auto activate(Layer const& prevLayer,  //
                                std::function<auto(Float)->Float> const& activationFunction  //
                                ) noexcept -> bool;

    [[nodiscard]] auto update(Layer const& prevLayer,  //
                              Float learningRate,  //
                              MatrixX const& delta  //
                              ) noexcept -> bool;

    size_t size() const noexcept;

public:
    MatrixX weights{};
    MatrixX biases{};
    MatrixX values{};
};

}  // namespace impl
