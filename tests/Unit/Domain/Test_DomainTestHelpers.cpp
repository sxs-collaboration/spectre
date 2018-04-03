// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>

#include "Domain/Side.hpp"
#include "Utilities/StdHelpers.hpp"  // IWYU pragma: keep
#include "tests/Unit/Domain/DomainTestHelpers.hpp"

namespace {
void test_vci_1d() {
  VolumeCornerIterator<1> vci{};
  CHECK(vci);
  CHECK(vci() == std::array<Side, 1>{{Side::Lower}});
  CHECK(vci.coords_of_corner() == std::array<double, 1>{{-1.0}});
  ++vci;
  CHECK(vci() == std::array<Side, 1>{{Side::Upper}});
  CHECK(vci.coords_of_corner() == std::array<double, 1>{{1.0}});
  ++vci;
  CHECK(not vci);
}

void test_vci_2d() {
  VolumeCornerIterator<2> vci{};
  CHECK(vci);
  CHECK(vci() == std::array<Side, 2>{{Side::Lower, Side::Lower}});
  CHECK(vci.coords_of_corner() == std::array<double, 2>{{-1.0, -1.0}});
  ++vci;
  CHECK(vci() == std::array<Side, 2>{{Side::Upper, Side::Lower}});
  CHECK(vci.coords_of_corner() == std::array<double, 2>{{1.0, -1.0}});
  ++vci;
  CHECK(vci() == std::array<Side, 2>{{Side::Lower, Side::Upper}});
  CHECK(vci.coords_of_corner() == std::array<double, 2>{{-1.0, 1.0}});
  ++vci;
  CHECK(vci() == std::array<Side, 2>{{Side::Upper, Side::Upper}});
  CHECK(vci.coords_of_corner() == std::array<double, 2>{{1.0, 1.0}});
  ++vci;
  CHECK(not vci);
}

void test_vci_3d() {
  VolumeCornerIterator<3> vci{};
  CHECK(vci);
  CHECK(vci() == std::array<Side, 3>{{Side::Lower, Side::Lower, Side::Lower}});
  CHECK(vci.coords_of_corner() == std::array<double, 3>{{-1.0, -1.0, -1.0}});
  ++vci;
  CHECK(vci() == std::array<Side, 3>{{Side::Upper, Side::Lower, Side::Lower}});
  CHECK(vci.coords_of_corner() == std::array<double, 3>{{1.0, -1.0, -1.0}});
  ++vci;
  CHECK(vci() == std::array<Side, 3>{{Side::Lower, Side::Upper, Side::Lower}});
  CHECK(vci.coords_of_corner() == std::array<double, 3>{{-1.0, 1.0, -1.0}});
  ++vci;
  CHECK(vci() == std::array<Side, 3>{{Side::Upper, Side::Upper, Side::Lower}});
  CHECK(vci.coords_of_corner() == std::array<double, 3>{{1.0, 1.0, -1.0}});
  ++vci;
  CHECK(vci() == std::array<Side, 3>{{Side::Lower, Side::Lower, Side::Upper}});
  CHECK(vci.coords_of_corner() == std::array<double, 3>{{-1.0, -1.0, 1.0}});
  ++vci;
  CHECK(vci() == std::array<Side, 3>{{Side::Upper, Side::Lower, Side::Upper}});
  CHECK(vci.coords_of_corner() == std::array<double, 3>{{1.0, -1.0, 1.0}});
  ++vci;
  CHECK(vci() == std::array<Side, 3>{{Side::Lower, Side::Upper, Side::Upper}});
  CHECK(vci.coords_of_corner() == std::array<double, 3>{{-1.0, 1.0, 1.0}});
  ++vci;
  CHECK(vci() == std::array<Side, 3>{{Side::Upper, Side::Upper, Side::Upper}});
  CHECK(vci.coords_of_corner() == std::array<double, 3>{{1.0, 1.0, 1.0}});
  ++vci;
  CHECK(not vci);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.TestHelpers.VolumeCornerIterator",
                  "[Domain][Unit]") {
  test_vci_1d();
  test_vci_2d();
  test_vci_3d();
}
