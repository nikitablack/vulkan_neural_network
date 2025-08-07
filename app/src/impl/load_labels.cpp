#include <cmrc/cmrc.hpp>
#include <cstring>
#include <impl/load_labels.hpp>
#include <impl/swap_endian.hpp>

CMRC_DECLARE(cmrc_dataset);

namespace impl {

auto load_labels(std::string const& name) noexcept -> std::vector<uint8_t> {
    auto const fs{cmrc::cmrc_dataset::get_filesystem()};

    auto const labelsRaw{fs.open(name)};

    uint32_t magicNumber{0};
    std::memcpy(&magicNumber, labelsRaw.begin(), sizeof(magicNumber));
    magicNumber = swap_endian(magicNumber);

    uint32_t labelCount{0};
    std::memcpy(&labelCount, labelsRaw.begin() + 4, sizeof(labelCount));
    labelCount = swap_endian(labelCount);

    std::vector<uint8_t> labels(labelCount);
    std::memcpy(labels.data(), labelsRaw.begin() + 8, labelCount);

    return labels;
}

}  // namespace impl
