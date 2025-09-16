#pragma once

#include <common/Float.hpp>
#include <vector>

namespace common {

template <typename T>
class LCG;

}

namespace impl {

class Neuron {
public:
    Neuron() = default;
    Neuron(size_t inputCount, common::LCG<common::Float>& lcg) noexcept;

public:
    common::Float value{0.0};
    common::Float bias{0.0};
    std::vector<common::Float> weights{};
};

}  // namespace impl
