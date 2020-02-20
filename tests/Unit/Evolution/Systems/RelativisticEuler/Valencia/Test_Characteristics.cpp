// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Direction.hpp"
#include "Domain/FaceNormal.hpp"
#include "Evolution/Systems/RelativisticEuler/Valencia/Characteristics.hpp"
#include "Evolution/Systems/RelativisticEuler/Valencia/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdHelpers.hpp"
#include "tests/Unit/DataStructures/DataBox/TestHelpers.hpp"
#include "tests/Unit/Domain/DomainTestHelpers.hpp"
#include "tests/Unit/PointwiseFunctions/Hydro/TestHelpers.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/Pypp.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Utilities/RandomUnitNormal.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace {

template <size_t Dim>
void test_compute_item_in_databox(
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, Dim>& shift,
    const tnsr::ii<DataVector, Dim>& spatial_metric,
    const tnsr::I<DataVector, Dim>& spatial_velocity,
    const Scalar<DataVector>& spatial_velocity_squared,
    const Scalar<DataVector>& sound_speed_squared,
    const tnsr::i<DataVector, Dim>& normal) noexcept {
  TestHelpers::db::test_compute_tag<
      RelativisticEuler::Valencia::Tags::CharacteristicSpeedsCompute<Dim>>(
      "CharacteristicSpeeds");
  const auto box = db::create<
      db::AddSimpleTags<
          gr::Tags::Lapse<>, gr::Tags::Shift<Dim>, gr::Tags::SpatialMetric<Dim>,
          hydro::Tags::SpatialVelocity<DataVector, Dim>,
          hydro::Tags::SoundSpeedSquared<DataVector>,
          ::Tags::Normalized<::Tags::UnnormalizedFaceNormal<Dim>>>,
      db::AddComputeTags<
          RelativisticEuler::Valencia::Tags::CharacteristicSpeedsCompute<Dim>>>(
      lapse, shift, spatial_metric, spatial_velocity, sound_speed_squared,
      normal);
  CHECK(RelativisticEuler::Valencia::characteristic_speeds(
            lapse, shift, spatial_velocity, spatial_velocity_squared,
            sound_speed_squared, normal) ==
        db::get<RelativisticEuler::Valencia::Tags::CharacteristicSpeeds<Dim>>(
            box));
}

template <size_t Dim>
void test_characteristic_speeds(const DataVector& used_for_size) noexcept {
  MAKE_GENERATOR(generator);
  namespace helper = hydro::TestHelpers;
  const auto nn_gen = make_not_null(&generator);
  const auto lapse = helper::random_lapse(nn_gen, used_for_size);
  const auto shift = helper::random_shift<Dim>(nn_gen, used_for_size);
  const auto spatial_metric =
      helper::random_spatial_metric<Dim>(nn_gen, used_for_size);
  const auto spatial_velocity = helper::random_velocity(
      nn_gen, helper::random_lorentz_factor(nn_gen, used_for_size),
      spatial_metric);
  const auto spatial_velocity_squared =
      dot_product(spatial_velocity, spatial_velocity, spatial_metric);
  EquationsOfState::PolytropicFluid<true> eos(0.001, 4.0 / 3.0);
  const auto rest_mass_density = helper::random_density(nn_gen, used_for_size);
  const Scalar<DataVector> sound_speed_squared{
      (get(eos.chi_from_density(rest_mass_density)) +
       get(eos.kappa_times_p_over_rho_squared_from_density(
           rest_mass_density))) /
      get(eos.specific_enthalpy_from_density(rest_mass_density))};

  // test with normal along coordinate axes
  for (const auto& direction : Direction<Dim>::all_directions()) {
    const auto normal = unit_basis_form(
        direction, determinant_and_inverse(spatial_metric).second);
    CHECK_ITERABLE_APPROX(
        RelativisticEuler::Valencia::characteristic_speeds(
            lapse, shift, spatial_velocity, spatial_velocity_squared,
            sound_speed_squared, normal),
        (pypp::call<std::array<DataVector, Dim + 2>>(
            "TestFunctions", "characteristic_speeds", lapse, shift,
            spatial_velocity, spatial_velocity_squared, sound_speed_squared,
            normal)));
  }

  // test with random normal
  const auto random_normal = raise_or_lower_index(
      random_unit_normal(nn_gen, spatial_metric), spatial_metric);
  CHECK_ITERABLE_APPROX(
      RelativisticEuler::Valencia::characteristic_speeds(
          lapse, shift, spatial_velocity, spatial_velocity_squared,
          sound_speed_squared, random_normal),
      (pypp::call<std::array<DataVector, Dim + 2>>(
          "TestFunctions", "characteristic_speeds", lapse, shift,
          spatial_velocity, spatial_velocity_squared, sound_speed_squared,
          random_normal)));
  test_compute_item_in_databox(lapse, shift, spatial_metric, spatial_velocity,
                               spatial_velocity_squared, sound_speed_squared,
                               random_normal);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.RelativisticEuler.Valencia.Characteristics",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/RelativisticEuler/Valencia"};

  GENERATE_UNINITIALIZED_DATAVECTOR;
  CHECK_FOR_DATAVECTORS(test_characteristic_speeds, (1, 2, 3));
}
