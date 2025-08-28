// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Domain/Structure/HasBoundary.hpp"
#include "Domain/Structure/Side.hpp"
#include "Domain/Structure/Topology.hpp"

namespace {
void test() {
  for (const auto side : std::array{Side::Lower, Side::Upper}) {
    CHECK(domain::has_boundary(domain::Topology::I1, side));
    CHECK_FALSE(domain::has_boundary(domain::Topology::S1, side));
    CHECK_FALSE(domain::has_boundary(domain::Topology::S2Colatitude, side));
    CHECK_FALSE(domain::has_boundary(domain::Topology::S2Longitude, side));
    CHECK_FALSE(domain::has_boundary(domain::Topology::B2Angular, side));
    CHECK_FALSE(domain::has_boundary(domain::Topology::B3Colatitude, side));
    CHECK_FALSE(domain::has_boundary(domain::Topology::B3Longitude, side));
    CHECK_FALSE(domain::has_boundary(domain::Topology::CartoonSphere, side));
  }
  CHECK_FALSE(domain::has_boundary(domain::Topology::B2Radial, Side::Lower));
  CHECK(domain::has_boundary(domain::Topology::B2Radial, Side::Upper));
  CHECK_FALSE(domain::has_boundary(domain::Topology::B3Radial, Side::Lower));
  CHECK(domain::has_boundary(domain::Topology::B3Radial, Side::Upper));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Structure.HasBoundary", "[Domain][Unit]") {
  test();
}
