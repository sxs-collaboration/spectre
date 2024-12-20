// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Domain/Structure/BlockGeometry.hpp"
#include "Utilities/GetOutput.hpp"

namespace domain {

SPECTRE_TEST_CASE("Unit.Domain.Structure.BlockGeometry", "[Domain][Unit]") {
  CHECK(get_output(domain::BlockGeometry::Cube) == "Cube");
  CHECK(get_output(domain::BlockGeometry::SphericalShell) == "SphericalShell");
}

}  // namespace domain
