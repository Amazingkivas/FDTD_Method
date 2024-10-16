#pragma once

#include "Structures.h"
#include "Field.h"

#include <iostream>
#include <fstream>
#include <string>

using namespace FDTDstruct;

void write_spherical(std::vector<Field> fields, Axis axis, std::string base_path, int it);
