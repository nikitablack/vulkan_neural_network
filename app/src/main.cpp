#include <fmt/core.h>

#include <cstdlib>
#include <impl/NeuralNetwork.hpp>
#include <impl/load_images.hpp>
#include <impl/load_labels.hpp>

namespace {

[[maybe_unused]] auto run_test_network() -> impl::NeuralNetwork {
    auto nn{impl::NeuralNetwork{std::vector<size_t>{2, 2, 2, 2}}};

    auto& layer0{nn.layers[0]};
    layer0.neurons[0].value = 1.0;
    layer0.neurons[1].value = 2.0;

    auto& layer1{nn.layers[1]};
    layer1.neurons[0].weights = {0.1, 0.1};
    layer1.neurons[0].bias = 0.1;
    layer1.neurons[1].weights = {0.2, 0.2};
    layer1.neurons[1].bias = 0.2;

    auto& layer2{nn.layers[2]};
    layer2.neurons[0].weights = {0.3, 0.3};
    layer2.neurons[0].bias = 0.3;
    layer2.neurons[1].weights = {0.4, 0.4};
    layer2.neurons[1].bias = 0.4;

    auto& layer3{nn.layers[3]};
    layer3.neurons[0].weights = {0.5, 0.5};
    layer3.neurons[0].bias = 0.5;
    layer3.neurons[1].weights = {0.6, 0.6};
    layer3.neurons[1].bias = 0.6;

    std::vector<std::vector<double>> input{{1.0, 2.0}};
    std::vector<double> output{};

    [[maybe_unused]] auto result{nn.train(input, {1}, 2, 0.1)};

    return nn;
}

}  // namespace

auto main(int /* argc */, char* /* argv */[]) -> int {
    // run_test_network();

    auto const labels{impl::load_labels("train-labels.idx1-ubyte")};
    auto const images{impl::load_images("train-images.idx3-ubyte")};

    if (labels.empty() || images.empty()) {
        fmt::println("Failed to load dataset labels or images.");
        return EXIT_FAILURE;
    }

    if (labels.size() != images.size()) {
        fmt::println("Mismatch between number of labels and images.");
        return EXIT_FAILURE;
    }

    fmt::println("Dataset size: {}.", labels.size());

    impl::NeuralNetwork nn{std::vector<size_t>{784, 8, 10}};
    size_t constexpr EPOCH_COUNT{10};
    double constexpr LEARNING_RATE{1.0};

    if (!nn.train(images, labels, EPOCH_COUNT, LEARNING_RATE)) {
        fmt::println("Failed to train neural network.");
        return EXIT_FAILURE;
    }

    // test
    {
        std::vector<double> output{};

        auto const testLabels{impl::load_labels("t10k-labels.idx1-ubyte")};
        auto const testImages{impl::load_images("t10k-images.idx3-ubyte")};

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
            auto predictedLabel{static_cast<uint8_t>(std::distance(output.begin(), maxIt))};

            if (predictedLabel == testLabels[i]) {
                ++correctCount;
            }
        }

        double const accuracy{static_cast<double>(correctCount) / testLabels.size()};
        fmt::println("Test accuracy: {:.2} ({}/{})", accuracy, correctCount, testLabels.size());
    }

    return EXIT_SUCCESS;
}