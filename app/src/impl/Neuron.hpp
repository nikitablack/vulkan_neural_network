#pragma once

#include <impl/Float.hpp>
#include <vector>

namespace impl {

class Neuron {
public:
    Neuron() = default;

    Neuron(Float valueArg, Float biasArg, size_t inputCountArg) noexcept;

public:
    Float value{0.0};
    Float bias{0.0};
    std::vector<Float> weights{};
};

}  // namespace impl
