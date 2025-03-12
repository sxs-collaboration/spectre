// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Domain/Structure/Topology.hpp"
#include "Utilities/GetOutput.hpp"

SPECTRE_TEST_CASE("Unit.Domain.Structure.Topology", "[Domain][Unit]") {
  CHECK(get_output(domain::Topology::Uninitialized) == "Uninitialized");
  CHECK(get_output(domain::Topology::I1) == "I1");
  CHECK(get_output(domain::Topology::S1) == "S1");
  CHECK(get_output(domain::Topology::S2Colatitude) == "S2Colatitude");
  CHECK(get_output(domain::Topology::S2Longitude) == "S2Longitude");
  CHECK(get_output(domain::Topology::B2Radial) == "B2Radial");
  CHECK(get_output(domain::Topology::B2Angular) == "B2Angular");
}
