// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "IO/Connectivity.hpp"
#include "Utilities/GetOutput.hpp"

SPECTRE_TEST_CASE("Unit.IO.Connectivity", "[Unit][IO][H5]") {
  CHECK(get_output(vis::detail::Topology::Line) == "Line"s);
  CHECK(get_output(vis::detail::Topology::Triangle) == "Triangle"s);
  CHECK(get_output(vis::detail::Topology::Quad) == "Quad"s);
  CHECK(get_output(vis::detail::Topology::Wedge) == "Wedge"s);
  CHECK(get_output(vis::detail::Topology::Hexahedron) == "Hexahedron"s);

  CHECK(vis::detail::xdmf_topology_type(vis::detail::Topology::Line) == 2);
  CHECK(vis::detail::xdmf_topology_type(vis::detail::Topology::Triangle) == 4);
  CHECK(vis::detail::xdmf_topology_type(vis::detail::Topology::Quad) == 5);
  CHECK(vis::detail::xdmf_topology_type(vis::detail::Topology::Wedge) == 8);
  CHECK(vis::detail::xdmf_topology_type(vis::detail::Topology::Hexahedron) ==
        9);
}
