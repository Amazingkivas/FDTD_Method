#include <vector>

#define C 3e10

class Field
{
private:
    int Ni;
    int Nj;
    std::vector<double> field;

public:
    Field(const int, const int);

    Field& operator= (const Field& other);

    double& operator() (int _i, int _j) { return field[_i * Nj + _j]; }

    int get_Ni() { return Ni; }
    int get_Nj() { return Nj; }
};

enum Component { EX, EY, EZ, BX, BY, BZ };

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
