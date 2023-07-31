// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <limits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/CurvedScalarWave/Characteristics.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Characteristics.hpp"
#include "Evolution/Systems/ScalarTensor/Characteristics.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"

namespace {
void check_max_char_speed(const DataVector& used_for_size) {
  MAKE_GENERATOR(gen);

  const Scalar<DataVector> gamma_1{used_for_size.size(), 0.0};
  const auto lapse = TestHelpers::gr::random_lapse(&gen, used_for_size);
  const auto shift = TestHelpers::gr::random_shift<3>(&gen, used_for_size);
  const auto spatial_metric =
      TestHelpers::gr::random_spatial_metric<3>(&gen, used_for_size);
  const Scalar<DataVector> gamma_1_scalar{used_for_size.size(), 0.0};

  double max_char_speed = std::numeric_limits<double>::signaling_NaN();
  ScalarTensor::Tags::ComputeLargestCharacteristicSpeed<
      Frame::Inertial>::function(make_not_null(&max_char_speed), gamma_1, lapse,
                                 shift, spatial_metric, gamma_1_scalar);

  double gh_max_char_speed = std::numeric_limits<double>::signaling_NaN();
  gh::Tags::ComputeLargestCharacteristicSpeed<3, Frame::Inertial>::function(
      make_not_null(&gh_max_char_speed), gamma_1, lapse, shift, spatial_metric);

  double scalar_max_char_speed = std::numeric_limits<double>::signaling_NaN();
  CurvedScalarWave::Tags::ComputeLargestCharacteristicSpeed<3>::function(
      make_not_null(&scalar_max_char_speed), gamma_1_scalar, lapse, shift,
      spatial_metric);

  CHECK(max_char_speed == std::max(gh_max_char_speed, scalar_max_char_speed));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ScalarTensor.MaxCharSpeed",
                  "[Unit][Evolution]") {
  check_max_char_speed(DataVector(5));
}
