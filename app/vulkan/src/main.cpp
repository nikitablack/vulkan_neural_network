#include <fmt/ranges.h>

#include <algorithm>
#include <common/LCG.hpp>
#include <common/Timer.hpp>
#include <common/load_images.hpp>
#include <common/load_labels.hpp>
#include <cstdlib>
#include <impl/NeuralNetwork.hpp>
#include <impl/graphics/GraphicsManager.hpp>

auto main(int /* argc */, char* /* argv */[]) -> int {
    common::LCG<float> lcg{42};

    impl::graphics::GraphicsManager graphicsManager{};
    graphicsManager.init();

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

    size_t constexpr TRAIN_COUNT{1};
    common::Timer trainTimer{};
    double totalTrainTimeMs{0.0};

    for (size_t t{0}; t < TRAIN_COUNT; ++t) {
        impl::NeuralNetwork nn{graphicsManager, {784, 1000, 1000, 10}, lcg};

        size_t constexpr EPOCH_COUNT{20};
        float constexpr LEARNING_RATE{1.0f};

        trainTimer.start();
        // can throw
        nn.train(graphicsManager, images, labels, EPOCH_COUNT, LEARNING_RATE);
        totalTrainTimeMs += trainTimer.stop();

        graphicsManager.flush();

        // test
        {
            std::vector<float> output{};

            auto const testLabels{common::load_labels("t10k-labels.idx1-ubyte")};
            auto const testImages{common::load_images("t10k-images.idx3-ubyte")};

            if (testLabels.size() != testImages.size()) {
                fmt::println("Mismatch between number of test labels and images.");
                return EXIT_FAILURE;
            }

            size_t correctCount{0};
            for (size_t i{0}; i < testLabels.size(); ++i) {
                // can throw
                nn.infer(graphicsManager, testImages[i], output);

                auto const maxIt{std::max_element(output.begin(), output.end())};
                auto const predictedLabel{static_cast<uint8_t>(std::distance(output.begin(), maxIt))};

                if (predictedLabel == testLabels[i]) {
                    ++correctCount;
                }
            }

            double const accuracy{static_cast<double>(correctCount) / testLabels.size()};
            fmt::println("Test accuracy: {:.2} ({}/{})", accuracy, correctCount, testLabels.size());
        }

        nn.clear(graphicsManager);
    }

    graphicsManager.clear();

    return EXIT_SUCCESS;
}