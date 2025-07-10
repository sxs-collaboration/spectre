// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Utilities/Kokkos/TestKernel.hpp"

KOKKOS_FUNCTION
double kernel_square(double x) { return x * x; }
