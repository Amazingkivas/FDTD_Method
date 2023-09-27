#include <vector>
#include <cmath>

namespace FDTD_Const
{
    const double C = 3e10;
}

class Field
{
private:
    int Ni;
    int Nj;
    std::vector<double> field;

public:
    Field(const int, const int);

    Field& operator= (const Field& other);

    double& operator() (int _i, int _j) { return field[_i * static_cast<double>(Nj) + _j]; }

    int get_Ni() { return Ni; }
    int get_Nj() { return Nj; }
};

class Borders
{
private:
    int I_prev = 0;
    int I_next = 0;
    int J_prev = 0;
    int J_next = 0;
    int border_i;
    int border_j;

public:
    Borders(int _Ni, int _Nj) : border_i(_Ni), border_j(_Nj) {}

    void neighborhood(int, int);

    int i_next() { return I_next; }
    int i_prev() { return I_prev; }
    int j_next() { return J_next; }
    int j_prev() { return J_prev; }
};

enum class Component { EX, EY, EZ, BX, BY, BZ };

class FDTD
{
private:
    Field Ex, Ey, Ez, Bx, By, Bz;
    int Ni, Nj;
    double ax, bx, ay, by, dx, dy, dt;

public:
    FDTD(int size_grid[2], double size_x[2], double size_y[2], double _dt);

    Field& get_field(Component);

    void update_field(const double&);
};
