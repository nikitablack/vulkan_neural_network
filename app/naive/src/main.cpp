#include <fmt/core.h>
#include <fmt/ranges.h>

#include <common/LCG.hpp>
#include <common/Timer.hpp>
#include <common/load_images.hpp>
#include <common/load_labels.hpp>
#include <cstdlib>
#include <impl/NeuralNetwork.hpp>

namespace {

[[maybe_unused]] auto run_test_network() -> void {
    common::LCG<common::Float> lcg{42};

    std::vector<float> input(784);
    std::vector<float> out(10);

    impl::NeuralNetwork nn2{{input.size(), 100, out.size()}, lcg};

    lcg = common::LCG<common::Float>{42};
    for (size_t i{1}; i < nn2.layers.size(); ++i) {
        for (auto& neuron : nn2.layers[i].neurons) {
            for (size_t w{0}; w < neuron.weights.size(); ++w) {
                neuron.weights[w] = lcg.next();
            }
        }

        for (auto& neuron : nn2.layers[i].neurons) {
            neuron.bias = lcg.next();
        }
    }

    [[maybe_unused]] auto r{nn2.forward(input, out)};

    fmt::println("{}", out);
}

}  // namespace

auto main(int /* argc */, char* /* argv */[]) -> int {
    run_test_network();
    return EXIT_SUCCESS;

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

    size_t constexpr LCG_SEED{42};
    common::LCG<common::Float> lcg{LCG_SEED};

    size_t constexpr TRAIN_COUNT{1};
    common::Timer trainTimer{};
    double totalTrainTimeMs{0.0};

    for (size_t t{0}; t < TRAIN_COUNT; ++t) {
        impl::NeuralNetwork nn{std::vector<size_t>{784, 100, 10}, lcg};

        // Reinitialize weights and biases.
        // We want that first all the weights are initialized, then all the biases for a layer, then move to
        // the next layer.
        //
        // However, the Neuron class first initialized only weights for itself and one bios after, then
        // the next neuron repeats the process. Only then all neurons are initialized, we move to the next layer.
        //
        // In psudocode:
        // for each Layer:
        //     for each Neuron:
        //         set neuron weights
        //         set neuron bias
        //
        // This breaks the comparison of results from another implementations, where first all the weights are
        // initialized and then all the biases.
        //
        // In psudocode:
        // for each Layer:
        //         set layer weights
        //         set layer biases
        lcg = common::LCG<common::Float>{LCG_SEED};
        for (size_t i{1}; i < nn.layers.size(); ++i) {
            for (auto& neuron : nn.layers[i].neurons) {
                for (size_t w{0}; w < neuron.weights.size(); ++w) {
                    neuron.weights[w] = lcg.next();
                }
            }

            for (auto& neuron : nn.layers[i].neurons) {
                neuron.bias = lcg.next();
            }
        }

        size_t constexpr EPOCH_COUNT{20};
        common::Float constexpr LEARNING_RATE{1.0};

        trainTimer.start();
        if (!nn.train(images, labels, EPOCH_COUNT, LEARNING_RATE)) {
            fmt::println("Failed to train neural network.");
            return EXIT_FAILURE;
        }
        totalTrainTimeMs += trainTimer.stop();

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
                auto predictedLabel{static_cast<uint8_t>(std::distance(output.begin(), maxIt))};

                if (predictedLabel == testLabels[i]) {
                    ++correctCount;
                }
            }

            double const accuracy{static_cast<double>(correctCount) / testLabels.size()};
            fmt::println("Test accuracy: {:.2} ({}/{})", accuracy, correctCount, testLabels.size());
        }
    }

    fmt::println("Average training time: {:.2f} ms", totalTrainTimeMs / TRAIN_COUNT);

    return EXIT_SUCCESS;
}