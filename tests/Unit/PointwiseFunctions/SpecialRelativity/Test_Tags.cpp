// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "DataStructures/Tensor/IndexType.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/SpecialRelativity/Tags.hpp"

class DataVector;

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.Hydro.Tags", "[Unit][Hydro]") {
  CHECK(sr::Tags::LorentzFactor<DataVector>::name() == "LorentzFactor");
  CHECK(sr::Tags::LorentzFactorSquared<DataVector>::name() ==
        "LorentzFactorSquared");
  /// [prefix_example]
  CHECK(sr::Tags::SpatialVelocity<DataVector, 3, Frame::Inertial>::name() ==
        "SpatialVelocity");
  CHECK(sr::Tags::SpatialVelocity<DataVector, 3, Frame::Grid>::name() ==
        "Grid_SpatialVelocity");
  CHECK(sr::Tags::SpatialVelocityOneForm<DataVector, 3,
                                         Frame::Inertial>::name() ==
        "SpatialVelocityOneForm");
  CHECK(
      sr::Tags::SpatialVelocityOneForm<DataVector, 3, Frame::Logical>::name() ==
      "Logical_SpatialVelocityOneForm");
  /// [prefix_example]
  CHECK(sr::Tags::SpatialVelocitySquared<double>::name() ==
        "SpatialVelocitySquared");
  CHECK(sr::Tags::SpatialVelocitySquared<DataVector>::name() ==
        "SpatialVelocitySquared");
}
