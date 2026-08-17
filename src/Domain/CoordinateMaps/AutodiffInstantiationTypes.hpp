// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Utilities/Autodiff/Autodiff.hpp"

#ifdef SPECTRE_AUTODIFF
#define MAP_AUTODIFF_TYPES \
  (autodiff::SecondOrderDual, autodiff::SecondOrderDualNum)
#else
#define MAP_AUTODIFF_TYPES
#endif
