// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <random>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Direction.hpp"
#include "Domain/FaceNormal.hpp"
#include "Evolution/Systems/RelativisticEuler/Valencia/Characteristics.hpp"
#include "Evolution/Systems/RelativisticEuler/Valencia/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdHelpers.hpp"
#include "tests/Unit/Domain/DomainTestHelpers.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/Pypp.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

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
  CHECK(RelativisticEuler::Valencia::Tags::CharacteristicSpeedsCompute<
            Dim>::name() == "CharacteristicSpeeds");
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
  //  Arbitrary random numbers can produce a negative radicand in Lambda^\pm.
  //  This bound helps to prevent that situation.
  const double max_value = 1.0 / sqrt(Dim);
  std::array<DataVector, Dim + 2> (*f)(
      const Scalar<DataVector>& lapse, const tnsr::I<DataVector, Dim>& shift,
      const tnsr::I<DataVector, Dim>& spatial_velocity,
      const Scalar<DataVector>& spatial_velocity_squared,
      const Scalar<DataVector>& sound_speed_squared,
      const tnsr::i<DataVector, Dim>& normal) =
      &RelativisticEuler::Valencia::characteristic_speeds<Dim>;
  pypp::check_with_random_values<6>(f, "TestFunctions", "characteristic_speeds",
                                    {{{0.0, 1.0},
                                      {-1.0, 1.0},
                                      {-max_value, max_value},
                                      {0.0, 1.0},
                                      {0.0, 1.0},
                                      {-max_value, max_value}}},
                                    used_for_size);
}

template <size_t Dim>
void test_with_normal_along_coordinate_axes(
    const DataVector& used_for_size) noexcept {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(0.0, 0.3);

  const auto nn_generator = make_not_null(&generator);
  const auto nn_distribution = make_not_null(&distribution);

  const auto lapse = make_with_random_values<Scalar<DataVector>>(
      nn_generator, nn_distribution, used_for_size);
  const auto shift = make_with_random_values<tnsr::I<DataVector, Dim>>(
      nn_generator, nn_distribution, used_for_size);
  const auto spatial_metric =
      make_with_random_values<tnsr::ii<DataVector, Dim>>(
          nn_generator, nn_distribution, used_for_size);
  const auto spatial_velocity =
      make_with_random_values<tnsr::I<DataVector, Dim>>(
          nn_generator, nn_distribution, used_for_size);
  const auto spatial_velocity_squared =
      dot_product(spatial_velocity, spatial_velocity, spatial_metric);
  const auto sound_speed_squared = make_with_random_values<Scalar<DataVector>>(
      nn_generator, nn_distribution, used_for_size);

  for (const auto& direction : Direction<Dim>::all_directions()) {
    const auto normal = euclidean_basis_vector(direction, used_for_size);

    CHECK_ITERABLE_APPROX(
        RelativisticEuler::Valencia::characteristic_speeds(
            lapse, shift, spatial_velocity, spatial_velocity_squared,
            sound_speed_squared, normal),
        (pypp::call<std::array<DataVector, Dim + 2>>(
            "TestFunctions", "characteristic_speeds", lapse, shift,
            spatial_velocity, spatial_velocity_squared, sound_speed_squared,
            normal)));
  }

  // test compute item with random normal vector
  test_compute_item_in_databox(
      lapse, shift, spatial_metric, spatial_velocity, spatial_velocity_squared,
      sound_speed_squared,
      make_with_random_values<tnsr::i<DataVector, Dim>>(
          nn_generator, nn_distribution, used_for_size));
}

}  // namespace

SPECTRE_TEST_CASE("Unit.RelativisticEuler.Valencia.Characteristics",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/RelativisticEuler/Valencia"};

  GENERATE_UNINITIALIZED_DATAVECTOR;
  CHECK_FOR_DATAVECTORS(test_characteristic_speeds, (1, 2, 3))
  // Test with aligned normals to check the code works
  // with vector components being 0.
  CHECK_FOR_DATAVECTORS(test_with_normal_along_coordinate_axes, (1, 2, 3))
}
