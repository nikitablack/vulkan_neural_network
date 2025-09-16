#include <fmt/core.h>

#include <common/LCG.hpp>
#include <impl/Layer.hpp>

namespace impl {

Layer::Layer(size_t neuronCount, size_t inputCount, common::LCG<common::Float>& lcg) noexcept
    : weights{neuronCount, inputCount}, biases{neuronCount, 1}, values{neuronCount, 1}, delta{neuronCount, 1} {
    if (inputCount > 0) {
        for (Eigen::Index r{0}; r < weights.rows(); ++r) {
            for (Eigen::Index c{0}; c < weights.cols(); ++c) {
                weights(r, c) = lcg.next();
            }
        }

        for (Eigen::Index r{0}; r < biases.rows(); ++r) {
            biases(r, 0) = lcg.next();
        }
    }
}

[[nodiscard]] auto Layer::activate(Layer const& prevLayer,  //
                                   std::function<auto(common::Float)->common::Float> const& activationFunction  //
                                   ) noexcept -> bool {
    if (weights.cols() != prevLayer.values.rows()) {
        fmt::println("Mismatch between values size and weights size.");
        return false;
    }

    values.noalias() = weights * prevLayer.values;
    values += biases;

    values = values.unaryExpr(activationFunction);

    return true;
}

[[nodiscard]] auto Layer::update(Layer const& prevLayer,  //
                                 common::Float learningRate  //
                                 ) noexcept -> bool {
    MatrixX const gradient{delta * prevLayer.values.transpose()};

    weights -= learningRate * gradient;
    biases -= learningRate * delta;

    return true;
}

size_t Layer::size() const noexcept {
    return static_cast<size_t>(weights.rows());
}

}  // namespace impl
