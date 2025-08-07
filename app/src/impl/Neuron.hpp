#pragma once

#include <vector>

namespace impl {

class Neuron {
public:
    Neuron() = default;

    Neuron(double valueArg, double biasArg, size_t inputCountArg) noexcept;

public:
    double value{0.0};
    double bias{0.0};
    std::vector<double> weights{};
};

}  // namespace impl
