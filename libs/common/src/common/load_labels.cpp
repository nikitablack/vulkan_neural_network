#include <cmrc/cmrc.hpp>
#include <common/load_labels.hpp>
#include <common/swap_endian.hpp>
#include <cstring>

CMRC_DECLARE(cmrc_dataset);

namespace common {

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

}  // namespace common
