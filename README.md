# Vulkan Neural Network (vknn)

This project implements a simple feedforward neural network for the MNIST dataset, with the following variants:
- **Naive C++ implementation - strightforward implementation using standard library and some sort of OOP** (`app/naive`)
- **Eigen-optimized implementation - linear-algebra-friendly implementation using Eigen library** (`app/eigen`)

All variants load the MNIST dataset from embedded resources (embedded with `cmakerc` library) and train/test a neural network with specified number of hidden layer.

---

## Requirements

- **Minimum CMake:** 3.25
- **Compilers tested:**
    - `gcc (Ubuntu 10.5.0-1ubuntu1~20.04) 10.5.0`
- **Internet connection** (required for CMake `FetchContent`)

> If you use a different version or compiler and encounter a compilation error, please let me know and I'll fix it as soon as possible.

---

## Building

```sh
cd vulkan_neural_network
mkdir build
cmake -S . -B build
cmake --build build
```

> There may be compilation warnings from some third-party libraries. You can ignore them.

## Configure options

- `VKNN_BUILD_EIGEN` - build Eigen-based implementation - `eigen_nn`, `ON` by default
    - `VKNN_USE_AVX2` - use `avx2` instructions set for the Eigen-based implementation, `OFF` by default, requires a CPU with avx2 support
- `VKNN_BUILD_NAIVE` - build naive implementation - `naive_nn`, `ON` by default
- `VKNN_USE_DOUBLE` - use double precision (`double`) data type instead of single precision (`float`), `OFF` by default

To use an option, provide it during the configuration step, for example:

```sh
cmake -S . -B build -DVKNN_USE_DOUBLE=ON -DVKNN_BUILD_EIGEN=OFF
```

---

## Project Structure

- `app/naive/` — Naive neural network implementation
- `app/eigen/` — Eigen-based neural network implementation
- `libs/common/` — Common utilities (e.g., data loading, timers)
- `libs/warnings/` — Compiler warning settings
- `resources/` — MNIST dataset files
- `third_party/` — External dependencies (fmt, cmakerc, eigen)

## Running

```sh
# naive implementation
./build/app/naive/naive_nn

# eigen implementation
./build/app/eigen/eigen_nn
```

All implementations will train and test a neural network on the embedded MNIST dataset.

## Dataset

The following MNIST files are embedded as resources:

- `train-images.idx3-ubyte`
- `train-labels.idx1-ubyte`
- `t10k-images.idx3-ubyte`
- `t10k-labels.idx1-ubyte`

## License

See [LICENSE](LICENSE).

## Credits

- [cmakerc](third_party/cmakerc)
- [Eigen](third_party/eigen)
- [fmt](third_party/fmt)