#ifndef TOWER_STATE_H
#define TOWER_STATE_H

#include <cstdint>

constexpr uint64_t TURRET_STATE_TIMEOUT_MS = 300;

struct TurretState
{
    double az_deg = 0.0;
    double el_deg = 0.0;
    uint64_t last_update_ms = 0;
    bool fresh = false;
};

TurretState getTurretState();
void setTurretState(double az_deg, double el_deg);
bool turretStateFresh(uint64_t timeout_ms = TURRET_STATE_TIMEOUT_MS);

#endif // TOWER_STATE_H
