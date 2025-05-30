#pragma once

#include "shared.h"
#include "Structures.h"
#include "FDTD.h"

using namespace FDTD_struct;

namespace FDTD_openmp {

using Boundaries = std::pair<int, int>;

class FDTD_PML : public FDTD {
private:
    Field Exy;
    Field Exz;
    Field Eyx;
    Field Eyz;
    Field Ezx;
    Field Ezy;

    Field Bxy;
    Field Bxz;
    Field Byx;
    Field Byz;
    Field Bzx;
    Field Bzy;

    Field EsigmaX;
    Field EsigmaY;
    Field EsigmaZ;

    Field BsigmaX;
    Field BsigmaY;
    Field BsigmaZ;

    Boundaries size_i_main, size_j_main, size_k_main;
    Boundaries size_i_solid, size_j_solid, size_k_solid;
    Boundaries size_i_part_from_start, size_i_part_from_end,
        size_k_part_from_start, size_k_part_from_end,
        size_xy_lower_k_pml, size_xy_upper_k_pml,
        size_yz_lower_i_pml, size_yz_upper_i_pml,
        size_zx_lower_j_pml, size_zx_upper_j_pml;

    int pml_size_i, pml_size_j, pml_size_k;

    inline void set_sigma_x(Boundaries bounds_i, Boundaries bounds_j, Boundaries bounds_k,
        double SGm, Function dist);
    inline void set_sigma_y(Boundaries bounds_i, Boundaries bounds_j, Boundaries bounds_k,
        double SGm, Function dist);
    inline void set_sigma_z(Boundaries bounds_i, Boundaries bounds_j, Boundaries bounds_k,
        double SGm, Function dist);

    inline double PMLcoef(double sigma) const;

    inline void update_E_PML(Boundaries bounds_i, Boundaries bounds_j, Boundaries bounds_k);

    inline void update_B_PML(Boundaries bounds_i, Boundaries bounds_j, Boundaries bounds_k);

public:
    FDTD_PML(Parameters _parameters, double _dt, double pml_percent);

    void update_fields() override;
};

}
