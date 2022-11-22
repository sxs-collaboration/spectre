// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <limits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/ForceFree/Characteristics.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
#include "Utilities/Gsl.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ForceFree.Characteristics",
                  "[Unit][Evolution]") {
  // test for tags
  TestHelpers::db::test_compute_tag<
      ForceFree::Tags::LargestCharacteristicSpeedCompute>(
      "LargestCharacteristicSpeed");

  // test for randomly generated fields
  MAKE_GENERATOR(gen);
  const DataVector used_for_size(5);

  const auto lapse =
      TestHelpers::gr::random_lapse(make_not_null(&gen), used_for_size);
  const auto shift =
      TestHelpers::gr::random_shift<3>(make_not_null(&gen), used_for_size);
  const auto spatial_metric = TestHelpers::gr::random_spatial_metric<3>(
      make_not_null(&gen), used_for_size);

  const auto shift_magnitude = magnitude(shift, spatial_metric);

  // Check that locally computed value matches the returned one
  double largest_characteristic_speed =
      std::numeric_limits<double>::signaling_NaN();
  ForceFree::Tags::LargestCharacteristicSpeedCompute::function(
      make_not_null(&largest_characteristic_speed), lapse, shift,
      spatial_metric);
  CHECK(largest_characteristic_speed == max(get(shift_magnitude) + get(lapse)));
}
