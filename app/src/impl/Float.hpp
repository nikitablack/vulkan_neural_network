#pragma once

namespace impl {

#ifdef USE_DOUBLE
using Float = double;
#else
using Float = float;
#endif

}  // namespace impl
