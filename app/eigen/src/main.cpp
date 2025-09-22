#include <fmt/core.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <common/LCG.hpp>
#include <common/load_images.hpp>
#include <common/load_labels.hpp>
#include <cstdlib>
#include <impl/NeuralNetwork.hpp>

namespace {

[[maybe_unused]] auto run_test_network() -> void {
    common::LCG<common::Float> lcg{42};

    auto nn{impl::NeuralNetwork{std::vector<size_t>{784, 100, 10}, lcg}};

    std::vector<common::Float> input(784);
    std::vector<common::Float> out{};
    [[maybe_unused]] auto r{nn.forward(input, out)};

    fmt::println("{}", out);
}

}  // namespace

auto main(int /* argc */, char* /* argv */[]) -> int {
    // run_test_network();
    // return EXIT_SUCCESS;

#if defined(EIGEN_VECTORIZE_AVX512)
    fmt::println("Using AVX-512");
#elif defined(EIGEN_VECTORIZE_AVX2)
    fmt::println("Using AVX2");
#elif defined(EIGEN_VECTORIZE_AVX)
    fmt::println("Using AVX\n");
#elif defined(EIGEN_VECTORIZE_SSE)
    fmt::println("Using SSE");
#else
    fmt::println("No vectorization");
#endif

    auto const labels{common::load_labels("train-labels.idx1-ubyte")};
    auto const images{common::load_images("train-images.idx3-ubyte")};

    if (labels.empty() || images.empty()) {
        fmt::println("Failed to load dataset labels or images.");
        return EXIT_FAILURE;
    }

    if (labels.size() != images.size()) {
        fmt::println("Mismatch between number of labels and images.");
        return EXIT_FAILURE;
    }

    common::LCG<common::Float> lcg{42};

    impl::NeuralNetwork nn{std::vector<size_t>{784, 100, 10}, lcg};

    size_t constexpr EPOCH_COUNT{20};
    common::Float constexpr LEARNING_RATE{1.0};

    if (!nn.train(images, labels, EPOCH_COUNT, LEARNING_RATE)) {
        fmt::println("Failed to train neural network.");
        return EXIT_FAILURE;
    }

    // test
    {
        std::vector<common::Float> output{};

        auto const testLabels{common::load_labels("t10k-labels.idx1-ubyte")};
        auto const testImages{common::load_images("t10k-images.idx3-ubyte")};

        if (testLabels.size() != testImages.size()) {
            fmt::println("Mismatch between number of test labels and images.");
            return EXIT_FAILURE;
        }

        size_t correctCount{0};
        for (size_t i{0}; i < testLabels.size(); ++i) {
            if (!nn.forward(testImages[i], output)) {
                fmt::println("Failed to compute forward pass for test image {}.", i);

                return EXIT_FAILURE;
            }

            auto const maxIt{std::max_element(output.begin(), output.end())};
            auto const predictedLabel{static_cast<uint8_t>(std::distance(output.begin(), maxIt))};

            if (predictedLabel == testLabels[i]) {
                ++correctCount;
            }
        }

        double const accuracy{static_cast<double>(correctCount) / testLabels.size()};
        fmt::println("Test accuracy: {:.2} ({}/{})", accuracy, correctCount, testLabels.size());
    }

    return EXIT_SUCCESS;
}