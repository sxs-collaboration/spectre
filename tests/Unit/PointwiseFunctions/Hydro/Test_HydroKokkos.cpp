// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <random>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Utilities/Kokkos/CheckOnDevice.hpp"
#include "PointwiseFunctions/Hydro/ComovingMagneticField.hpp"
#include "PointwiseFunctions/Hydro/TransportVelocity.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

namespace {
void test_comoving_magnetic_field(
    const gsl::not_null<std::mt19937*> generator) {
  std::uniform_real_distribution<> distribution(0.1, 1.0);
  const auto spatial_velocity = make_with_random_values<tnsr::I<double, 3>>(
      generator, make_not_null(&distribution));
  const auto magnetic_field = make_with_random_values<tnsr::I<double, 3>>(
      generator, make_not_null(&distribution));
  const auto spatial_velocity_one_form =
      make_with_random_values<tnsr::i<double, 3>>(generator,
                                                  make_not_null(&distribution));
  const auto magnetic_field_one_form =
      make_with_random_values<tnsr::i<double, 3>>(generator,
                                                  make_not_null(&distribution));
  const auto magnetic_field_squared = make_with_random_values<Scalar<double>>(
      generator, make_not_null(&distribution));
  const auto magnetic_field_dot_spatial_velocity =
      make_with_random_values<Scalar<double>>(generator,
                                              make_not_null(&distribution));
  const Scalar<double> lorentz_factor{1.5};
  const auto shift = make_with_random_values<tnsr::I<double, 3>>(
      generator, make_not_null(&distribution));
  const auto lapse = make_with_random_values<Scalar<double>>(
      generator, make_not_null(&distribution));
  TestHelpers::check_on_device(KOKKOS_LAMBDA() {
    return TestHelpers::flatten(
        hydro::comoving_magnetic_field(spatial_velocity, magnetic_field,
                                       magnetic_field_dot_spatial_velocity,
                                       lorentz_factor, shift, lapse),
        hydro::comoving_magnetic_field_one_form(
            spatial_velocity_one_form, magnetic_field_one_form,
            magnetic_field_dot_spatial_velocity, lorentz_factor, shift, lapse),
        hydro::comoving_magnetic_field_squared(
            magnetic_field_squared, magnetic_field_dot_spatial_velocity,
            lorentz_factor));
  });
}

void test_transport_velocity(const gsl::not_null<std::mt19937*> generator) {
  std::uniform_real_distribution<> distribution(0.1, 1.0);
  const auto spatial_velocity = make_with_random_values<tnsr::I<double, 3>>(
      generator, make_not_null(&distribution));
  const auto lapse = make_with_random_values<Scalar<double>>(
      generator, make_not_null(&distribution));
  const auto shift = make_with_random_values<tnsr::I<double, 3>>(
      generator, make_not_null(&distribution));
  TestHelpers::check_on_device(KOKKOS_LAMBDA() {
    // This function has no return-by-value overload
    tnsr::I<double, 3> transport_velocity{};
    hydro::transport_velocity(make_not_null(&transport_velocity),
                              spatial_velocity, lapse, shift);
    return TestHelpers::flatten(transport_velocity);
  });
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.Hydro.Kokkos",
                  "[Unit][PointwiseFunctions]") {
  MAKE_GENERATOR(generator);
  test_comoving_magnetic_field(make_not_null(&generator));
  test_transport_velocity(make_not_null(&generator));
}
