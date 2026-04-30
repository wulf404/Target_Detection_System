#include "turret_command.h"

#include "can_work.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace turret_control {
namespace {

constexpr double kAzLimitDeg = 170.0;
constexpr double kElLimitDeg = 80.0;
constexpr double kStepDeg = 0.01;
constexpr double kCentideg = 100.0;
constexpr canid_t kPositionCanId = 0x05;

double clampDeg(double value, double limit)
{
    return std::max(-limit, std::min(limit, value));
}

double roundToStep(double value)
{
    return std::round(value / kStepDeg) * kStepDeg;
}

std::uint16_t encodeCentideg(double value)
{
    const auto signed_value = static_cast<std::int16_t>(std::lround(value * kCentideg));
    return static_cast<std::uint16_t>(signed_value);
}

can_frame makePositionFrameWords(std::uint16_t az, std::uint16_t el, std::uint8_t control)
{
    can_frame frame{};
    frame.can_id = kPositionCanId;
    frame.can_dlc = 5;
    frame.data[0] = control;
    frame.data[1] = static_cast<std::uint8_t>((az >> 8) & 0xFF);
    frame.data[2] = static_cast<std::uint8_t>(az & 0xFF);
    frame.data[3] = static_cast<std::uint8_t>((el >> 8) & 0xFF);
    frame.data[4] = static_cast<std::uint8_t>(el & 0xFF);
    return frame;
}

} // namespace

AnglesDeg normalize(AnglesDeg angles)
{
    angles.az = roundToStep(clampDeg(angles.az, kAzLimitDeg));
    angles.el = roundToStep(clampDeg(angles.el, kElLimitDeg));
    return angles;
}

can_frame makePositionFrame(AnglesDeg angles, std::uint8_t control)
{
    angles = normalize(angles);

    const std::uint16_t az = encodeCentideg(angles.az);
    const std::uint16_t el = encodeCentideg(angles.el);

    return makePositionFrameWords(az, el, control);
}

can_frame makePositionFrameCentideg(std::int16_t az_centideg,
                                    std::int16_t el_centideg,
                                    std::uint8_t control)
{
    return makePositionFrameWords(static_cast<std::uint16_t>(az_centideg),
                                  static_cast<std::uint16_t>(el_centideg),
                                  control);
}

bool sendPosition(can_work* can, AnglesDeg angles, std::uint8_t control)
{
    if (!can) {
        return false;
    }

    can->writeCan(makePositionFrame(angles, control));
    return true;
}

} // namespace turret_control
