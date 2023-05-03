// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/HalfPiPhiTwoNormals.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"

namespace {
template <size_t SpatialDim, typename Frame>
void test_with_python(const DataVector& used_for_size) {
  pypp::check_with_random_values<1>(
      &gh::gauges::half_pi_and_phi_two_normals<SpatialDim, Frame>,
      "Evolution.Systems.GeneralizedHarmonic.GaugeSourceFunctions."
      "HalfPiPhiTwoNormals",
      {"half_pi_two_normals", "half_phi_two_normals"}, {{{-0.01, 0.01}}},
      used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GeneralizedHarmonic.Gauge.HalfPiPhiTwoNormals",
    "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{""};
  const DataVector used_for_size(5);

  test_with_python<1, Frame::Grid>(used_for_size);
  test_with_python<2, Frame::Grid>(used_for_size);
  test_with_python<3, Frame::Grid>(used_for_size);
  test_with_python<1, Frame::Inertial>(used_for_size);
  test_with_python<2, Frame::Inertial>(used_for_size);
  test_with_python<3, Frame::Inertial>(used_for_size);
}
