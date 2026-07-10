// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Helpers/Evolution/DiscontinuousGalerkin/Actions/ComputeTimeDerivativeImpl.tpp"

namespace TestHelpers::evolution::dg::Actions {
template void test<SystemType::Mixed, false, 2>();
template void test_LDG<SystemType::Mixed, 2>();
}  // namespace TestHelpers::evolution::dg::Actions
