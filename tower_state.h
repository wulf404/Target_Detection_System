#ifndef TOWER_STATE_H
#define TOWER_STATE_H

struct TurretState
{
    double az_deg = 0.0;
    double el_deg = 0.0;
};

TurretState getTurretState();
void setTurretState(double az_deg, double el_deg);

#endif // TOWER_STATE_H
