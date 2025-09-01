# Vulkan Neural Network (vknn)

This project implements a simple feedforward neural network for the MNIST dataset, with the following variants:
- **Naive C++ implementation** (`app/naive`)
- **Eigen-optimized implementation** (`app/eigen`)

All variants load the MNIST dataset from embedded resources and train/test a neural network with one hidden layer.

## Features

- C++17 codebase
- Modular structure with common utilities ([libs/common](libs/common))
- Strict warning and error handling ([libs/warnings](libs/warnings))
- Resource embedding using [cmakerc](third_party/cmakerc)
- Optional Eigen-based vectorization ([third_party/eigen](third_party/eigen))
- MNIST dataset embedded as resources ([resources](resources))

## Project Structure

- `app/naive/` — Naive (e.g. _brute-force_) neural network implementation
- `app/eigen/` — Eigen-based neural network implementation
- `libs/common/` — Common utilities (e.g., data loading, timers)
- `libs/warnings/` — Compiler warning settings
- `resources/` — MNIST dataset files
- `third_party/` — External dependencies (fmt, cmakerc, eigen)

## Building

This project uses CMake (>= 3.25):

```sh
cmake -S . -B build
cmake --build build
```

You can enable/disable implementations via CMake options:

- `BUILD_NAIVE` (default ON)
- `BUILD_EIGEN` (default ON)

Example:

```sh
cmake -S . -B build -DBUILD_NAIVE=ON -DBUILD_EIGEN=OFF
```

## Running

Executables are built in `build/app/naive/` and `build/app/eigen/`:

```sh
./build/app/naive/naive_nn
./build/app/eigen/eigen_nn
```

Both will train and test a neural network on the embedded MNIST dataset.

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