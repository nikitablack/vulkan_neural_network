#pragma once

#include <impl/Float.hpp>
#include <vector>

namespace impl {

class Neuron {
public:
    Neuron() = default;
    Neuron(size_t inputCount) noexcept;

public:
    Float value{0.0};
    Float bias{0.0};
    std::vector<Float> weights{};
};

}  // namespace impl
