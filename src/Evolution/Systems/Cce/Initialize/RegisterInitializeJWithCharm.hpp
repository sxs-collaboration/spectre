// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Systems/Cce/AnalyticSolutions/LinearizedBondiSachs.hpp"
#include "Evolution/Systems/Cce/Initialize/InitializeJ.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"

/// \cond
namespace Cce {
namespace Solutions::LinearizedBondiSachs_detail::InitializeJ {
struct LinearizedBondiSachs;
}
/// \endcond

/// A function for registering all of the InitializeJ derived classes with
/// charm, including the ones not intended to be directly option-creatable
template <bool uses_partially_flat_cartesian_coordinates>
void register_initialize_j_with_charm() {
  PUPable_reg(SINGLE_ARG(Solutions::LinearizedBondiSachs_detail::InitializeJ::
                         LinearizedBondiSachs));
  Parallel::register_derived_classes_with_charm<Cce::InitializeJ::InitializeJ<
      uses_partially_flat_cartesian_coordinates>>();
}
}  // namespace Cce
