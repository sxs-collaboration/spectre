// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Helpers/Evolution/DiscontinuousGalerkin/Actions/SystemType.hpp"

namespace TestHelpers::evolution::dg::Actions {
template <SystemType system_type, bool UsePrims, size_t Dim>
void test();

template <SystemType system_type, size_t Dim>
void test_LDG();
}  // namespace TestHelpers::evolution::dg::Actions
