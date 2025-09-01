#pragma once

namespace common {

#ifdef USE_DOUBLE
using Float = double;
#else
using Float = float;
#endif

constexpr Float operator"" _F(long double v) {
    return static_cast<Float>(v);
}

}  // namespace common
