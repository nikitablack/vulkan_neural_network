#pragma once

#include <common/Float.hpp>
#include <vector>

namespace impl {

class Neuron {
public:
    Neuron() = default;
    Neuron(size_t inputCount) noexcept;

public:
    common::Float value{0.0};
    common::Float bias{0.0};
    std::vector<common::Float> weights{};
};

}  // namespace impl
