// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp" // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Characteristics.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp" //IWYU pragma: keep
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"

// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_forward_declare Variables
// IWYU pragma: no_forward_declare Tags::CharSpeed
// IWYU pragma: no_forward_declare GeneralizedHarmonic::Tags::UPsi
// IWYU pragma: no_forward_declare GeneralizedHarmonic::Tags::UZero
// IWYU pragma: no_forward_declare GeneralizedHarmonic::Tags::UMinus
// IWYU pragma: no_forward_declare GeneralizedHarmonic::Tags::UPlus

namespace {
template <typename Tag, size_t Dim, typename Frame>
Scalar<DataVector> compute_speed_with_tag(
    const Scalar<DataVector>& gamma_1, const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame>& shift,
    const tnsr::i<DataVector, Dim, Frame>& normal) {
  return get<Tag>(
      GeneralizedHarmonic::CharacteristicSpeedsCompute<Dim, Frame>::function(
          gamma_1, lapse, shift, normal));
}

template <size_t Dim, typename Frame>
void test_characteristic_speeds() noexcept {
  const DataVector used_for_size(5);
  pypp::check_with_random_values<1>(
      compute_speed_with_tag<
          Tags::CharSpeed<GeneralizedHarmonic::Tags::UPsi<Dim, Frame>>, Dim,
          Frame>,
      "TestFunctions", "char_speed_upsi", {{{-10.0, 10.0}}}, used_for_size);
  pypp::check_with_random_values<1>(
      compute_speed_with_tag<
          Tags::CharSpeed<GeneralizedHarmonic::Tags::UZero<Dim, Frame>>, Dim,
          Frame>,
      "TestFunctions", "char_speed_uzero", {{{-10.0, 10.0}}}, used_for_size);
  pypp::check_with_random_values<1>(
      compute_speed_with_tag<
          Tags::CharSpeed<GeneralizedHarmonic::Tags::UMinus<Dim, Frame>>, Dim,
          Frame>,
      "TestFunctions", "char_speed_uminus", {{{-10.0, 10.0}}}, used_for_size);
  pypp::check_with_random_values<1>(
      compute_speed_with_tag<
          Tags::CharSpeed<GeneralizedHarmonic::Tags::UPlus<Dim, Frame>>, Dim,
          Frame>,
      "TestFunctions", "char_speed_uplus", {{{-10.0, 10.0}}}, used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.GeneralizedHarmonic.CharSpeeds",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/GeneralizedHarmonic/"};

  test_characteristic_speeds<1, Frame::Grid>();
  test_characteristic_speeds<2, Frame::Grid>();
  test_characteristic_speeds<3, Frame::Grid>();
  test_characteristic_speeds<1, Frame::Inertial>();
  test_characteristic_speeds<2, Frame::Inertial>();
  test_characteristic_speeds<3, Frame::Inertial>();
}
