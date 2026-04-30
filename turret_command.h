#pragma once

#include <cstdint>
#include <linux/can.h>

class can_work;

namespace turret_control {

struct AnglesDeg
{
    double az = 0.0;
    double el = 0.0;
};

AnglesDeg normalize(AnglesDeg angles);
can_frame makePositionFrame(AnglesDeg angles, std::uint8_t control = 192);
can_frame makePositionFrameCentideg(std::int16_t az_centideg,
                                    std::int16_t el_centideg,
                                    std::uint8_t control = 192);
bool sendPosition(can_work* can, AnglesDeg angles, std::uint8_t control = 192);

} // namespace turret_control
