#pragma once

#include <Eigen/Dense>
#include <functional>
#include <impl/Neuron.hpp>
#include <vector>

namespace impl {

class Layer {
public:
    Layer() noexcept = default;

    Layer(size_t neuronCount, size_t inputCount) noexcept;

public:
    [[nodiscard]] auto activate(Layer const& prevLayer,  //
                                std::function<auto(double)->double> const& activationFunction  //
                                ) noexcept -> bool;

    [[nodiscard]] auto update(Layer const& prevLayer,  //
                              double learningRate,  //
                              Eigen::MatrixXd const& delta  //
                              ) noexcept -> bool;

    size_t size() const noexcept;

public:
    Eigen::MatrixXd weights{};
    Eigen::MatrixXd biases{};
    Eigen::MatrixXd values{};
};

}  // namespace impl
