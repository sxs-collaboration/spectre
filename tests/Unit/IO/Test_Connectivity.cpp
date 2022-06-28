// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "IO/Connectivity.hpp"
#include "Utilities/GetOutput.hpp"

SPECTRE_TEST_CASE("Unit.IO.Connectivity", "[Unit][IO][H5]") {
  CHECK(get_output(vis::detail::Topology::Line) == "Line"s);
  CHECK(get_output(vis::detail::Topology::Quad) == "Quad"s);
  CHECK(get_output(vis::detail::Topology::Hexahedron) == "Hexahedron"s);
}
