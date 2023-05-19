// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/Tensor/IndexType.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

namespace {

void test_registration_name() {
  CHECK(registration_name<
            domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                                  domain::CoordinateMaps::ProductOf2Maps<
                                      domain::CoordinateMaps::Affine,
                                      domain::CoordinateMaps::Affine>>>() ==
        "domain::CoordinateMap<Frame::BlockLogical,Frame::Inertial,"
        "domain::CoordinateMaps::ProductOf2Maps<domain::CoordinateMaps::"
        "Affine,domain::CoordinateMaps::Affine>>");
}

}

SPECTRE_TEST_CASE("Unit.Parallel.RegisterDerivedClassesWithCharm",
                  "[Unit][Parallel]") {
  test_registration_name();
}
