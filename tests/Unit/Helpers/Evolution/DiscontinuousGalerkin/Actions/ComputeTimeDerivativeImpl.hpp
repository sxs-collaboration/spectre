// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Helpers/Evolution/DiscontinuousGalerkin/Actions/SystemType.hpp"

namespace TestHelpers::evolution::dg::Actions {
template <SystemType system_type, size_t Dim>
void test();
}  // namespace TestHelpers::evolution::dg::Actions
