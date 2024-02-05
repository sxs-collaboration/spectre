// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "IO/Exporter/Exporter.hpp"
#include "Informer/InfoFromBuild.hpp"

namespace spectre::Exporter {

SPECTRE_TEST_CASE("Unit.IO.Exporter", "[Unit]") {
  const auto interpolated_data = interpolate_to_points<3>(
      unit_test_src_path() + "/Visualization/Python/VolTestData*.h5",
      "element_data", 0, {"Psi", "Phi_x", "Phi_y", "Phi_z"},
      {{{0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, {0.0, 0.0, 0.0}}});
  const auto& psi = interpolated_data[0];
  CHECK(psi[0] == approx(-0.07059806932542323));
  CHECK(psi[1] == approx(0.7869554122196492));
  CHECK(psi[2] == approx(0.9876185584100299));
  const auto& phi_y = interpolated_data[2];
  CHECK(phi_y[0] == approx(1.0569673471948728));
  CHECK(phi_y[1] == approx(0.6741524090220188));
  CHECK(phi_y[2] == approx(0.2629752479142838));
}

}  // namespace spectre::Exporter
