// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/Matrix.hpp"

#include <pup.h>  // IWYU pragma: keep

#include "Parallel/PupStlCpp11.hpp"  // IWYU pragma: keep

void operator|(PUP::er& p, Matrix& matrix) { p | *matrix.data(); }
