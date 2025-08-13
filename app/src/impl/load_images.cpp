#include <cmrc/cmrc.hpp>
#include <cstring>
#include <impl/load_images.hpp>
#include <impl/swap_endian.hpp>

CMRC_DECLARE(cmrc_dataset);

namespace impl {

auto load_images(std::string const& name) noexcept -> std::vector<std::vector<Float>> {
    auto const fs{cmrc::cmrc_dataset::get_filesystem()};

    auto imagesRaw{fs.open(name)};

    uint32_t magicNumber{0};
    std::memcpy(&magicNumber, imagesRaw.begin(), sizeof(magicNumber));
    magicNumber = swap_endian(magicNumber);

    uint32_t imageCount{0};
    std::memcpy(&imageCount, imagesRaw.begin() + 4, sizeof(imageCount));
    imageCount = swap_endian(imageCount);

    uint32_t rows{0}, cols{0};
    std::memcpy(&rows, imagesRaw.begin() + 8, sizeof(rows));
    std::memcpy(&cols, imagesRaw.begin() + 12, sizeof(cols));
    rows = swap_endian(rows);
    cols = swap_endian(cols);

    std::vector<std::vector<uint8_t>> imagesByte(imageCount, std::vector<uint8_t>(rows * cols));
    for (size_t i{0}; i < imageCount; ++i) {
        std::memcpy(imagesByte[i].data(), imagesRaw.begin() + 16 + i * rows * cols, rows * cols);
    }

    std::vector<std::vector<Float>> images(imageCount, std::vector<Float>(rows * cols));
    for (size_t i{0}; i < imageCount; ++i) {
        for (size_t j{0}; j < rows * cols; ++j) {
            images[i][j] = static_cast<Float>(imagesByte[i][j]) / static_cast<Float>(255);  // Normalize to [0, 1]
        }
    }

    return images;
}

}  // namespace impl
