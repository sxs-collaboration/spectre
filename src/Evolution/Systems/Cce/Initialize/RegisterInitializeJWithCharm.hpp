// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Systems/Cce/Components/WorldtubeBoundary.hpp"
#include "Evolution/Systems/Cce/Initialize/InitializeJ.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

namespace Cce {
/// A function for registering all of the InitializeJ derived classes with
/// charm, including the ones not intended to be directly option-creatable.
/// `LinearizedBondiSachs` is registered through
/// `InitializeJ<false>::creatable_classes` (it sets `factory_creatable = false`
/// so it stays out of the option factory).
template <bool EvolveCcm, typename BoundaryComponent>
void register_initialize_j_with_charm() {
  if constexpr (tt::is_a_v<AnalyticWorldtubeBoundary, BoundaryComponent>) {
    register_derived_classes_with_charm<Cce::InitializeJ::InitializeJ<false>>();
  } else {
    register_derived_classes_with_charm<
        Cce::InitializeJ::InitializeJ<EvolveCcm>>();
  }
}
}  // namespace Cce
