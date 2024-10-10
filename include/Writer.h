#pragma once

#include "Structures.h"
#include "Field.h"

#include <iostream>
#include <fstream>
#include <string>

using namespace FDTDstruct;

void write_x(Field& this_field, std::ofstream& fout);
void write_y(Field& this_field, std::ofstream& fout);
void write_z(Field& this_field, std::ofstream& fout);

void write_spherical(std::vector<Field> fields, Axis axis, std::string base_path, int it);
