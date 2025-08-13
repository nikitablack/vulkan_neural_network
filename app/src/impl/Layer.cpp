#include <fmt/core.h>

#include <impl/Layer.hpp>
#include <random>

namespace impl {

Layer::Layer(size_t neuronCount, size_t inputCount) noexcept
    : weights{neuronCount, inputCount}, biases{neuronCount, 1}, values{neuronCount, 1} {
    if (inputCount > 0) {
        std::random_device rd{};
        std::mt19937 gen{rd()};
        std::uniform_real_distribution<Float> dist{-1.0, 1.0};

        for (Eigen::Index r{0}; r < weights.rows(); ++r) {
            for (Eigen::Index c{0}; c < weights.cols(); ++c) {
                weights(r, c) = dist(gen);
            }
        }

        for (Eigen::Index r{0}; r < biases.rows(); ++r) {
            biases(r, 0) = dist(gen);
        }
    }
}

[[nodiscard]] auto Layer::activate(Layer const& prevLayer,  //
                                   std::function<auto(Float)->Float> const& activationFunction  //
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
                                 Float learningRate,  //
                                 MatrixX const& delta  //
                                 ) noexcept -> bool {
    if (static_cast<size_t>(delta.rows()) != size()) {
        return false;  // Mismatch in delta size
    }

    MatrixX const gradient{delta * prevLayer.values.transpose()};

    weights -= learningRate * gradient;
    biases -= learningRate * delta;

    return true;
}

size_t Layer::size() const noexcept {
    return static_cast<size_t>(weights.rows());
}

}  // namespace impl
