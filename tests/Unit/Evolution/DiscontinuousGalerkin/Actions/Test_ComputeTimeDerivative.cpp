// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Helpers/Evolution/DiscontinuousGalerkin/Actions/ComputeTimeDerivativeImpl.hpp"

namespace TestHelpers::evolution::dg::Actions {
namespace {
template <SystemType system_type, UseBoundaryCorrection use_boundary_correction>
void test_wrapper() {
  test<system_type, use_boundary_correction, 1>();
  test<system_type, use_boundary_correction, 2>();
  test<system_type, use_boundary_correction, 3>();
}

SPECTRE_TEST_CASE("Unit.Evolution.DG.ComputeTimeDerivative",
                  "[Unit][Evolution][Actions]") {
  // The test is designed to test the `ComputeTimeDerivative` action for DG.
  // This action does a lot:
  //
  // - compute partial derivatives as needed
  // - compute the time derivative from
  //   `System::compute_volume_time_derivative`. This includes fluxes, sources,
  //   and nonconservative products.
  // - adds moving mesh terms as needed.
  // - compute flux divergence and add to the time derivative.
  // - compute mortar data for internal boundaries.
  //
  // The action supports conservative systems, and nonconservative systems
  // (mixed conservative-nonconservative systems will be added in the future).
  //
  // To test the action thoroughly we need to test a lot of different
  // combinations:
  //
  // - system type (conservative/nonconservative), using the enum SystemType
  // - 1d, 2d, 3d
  // - whether the mesh is moving or not
  //
  // Note that because the test is quite expensive to build, we have split the
  // compilation across multiple translation units by having the test be defined
  // in ComputeTimeDerivativeImpl.tpp.

  test_wrapper<SystemType::Nonconservative, UseBoundaryCorrection::No>();
  test_wrapper<SystemType::Conservative, UseBoundaryCorrection::No>();

  test_wrapper<SystemType::Conservative, UseBoundaryCorrection::Yes>();
  test_wrapper<SystemType::Nonconservative, UseBoundaryCorrection::Yes>();
  test_wrapper<SystemType::Mixed, UseBoundaryCorrection::Yes>();
}
}  // namespace
}  // namespace TestHelpers::evolution::dg::Actions
