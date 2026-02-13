// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Utilities/Autodiff/Autodiff.hpp"

#ifdef SPECTRE_AUTODIFF
#define MAP_AUTODIFF_TYPES                                  \
  (autodiff::SecondOrderDual, autodiff::SecondOrderDualNum, \
   std::reference_wrapper<const autodiff::SecondOrderDual>, \
   std::reference_wrapper<const autodiff::SecondOrderDualNum>)
#else
#define MAP_AUTODIFF_TYPES
#endif
