// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Helpers/Evolution/DiscontinuousGalerkin/Actions/SystemType.hpp"

namespace TestHelpers::evolution::dg::Actions {
// Used to select between the new style of boundary correction and the "boundary
// scheme" method.
enum UseBoundaryCorrection { Yes, No };

template <SystemType system_type, UseBoundaryCorrection use_boundary_correction,
          size_t Dim>
void test() noexcept;
}  // namespace TestHelpers::evolution::dg::Actions
