// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <iomanip>
#include <limits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Hydro/StressEnergy.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {
template <typename DataType>
void test_stress_trace(const DataType& velocity_squared) {
  using std::sqrt;
  const Scalar<DataType> spatial_velocity_squared{velocity_squared};
  const Scalar<DataType> lorentz_factor{1. / sqrt(1. - velocity_squared)};
  // A unit absolute scale would hide the loss of the entire kinetic term.
  const auto relative_approx =
      Approx::custom()
          .epsilon(128. * std::numeric_limits<double>::epsilon())
          .scale(0.);
  for (const double density : {0., 1.e-20, 2.}) {
    for (const double pressure_over_density : {0., 0.125}) {
      // A gamma = 3/2 gamma-law state with epsilon = 2 p/rho. At zero
      // density, use the finite limiting enthalpy and zero pressure.
      const auto rest_mass_density =
          make_with_value<Scalar<DataType>>(velocity_squared, density);
      const auto specific_enthalpy = make_with_value<Scalar<DataType>>(
          velocity_squared, 1. + 3. * pressure_over_density);
      const auto pressure = make_with_value<Scalar<DataType>>(
          velocity_squared, density * pressure_over_density);
      for (const double magnetic_field_squared : {0., 0.25}) {
        for (const bool parallel_field : {false, true}) {
          CAPTURE(density, pressure_over_density, magnetic_field_squared,
                  parallel_field, velocity_squared);
          const auto magnetic_field_dot_velocity =
              parallel_field
                  ? Scalar<DataType>{sqrt(magnetic_field_squared *
                                          velocity_squared)}
                  : make_with_value<Scalar<DataType>>(velocity_squared, 0.);
          const Scalar<DataType> comoving_magnetic_field_squared{
              magnetic_field_squared * (1. - velocity_squared) +
              get(magnetic_field_dot_velocity) *
                  get(magnetic_field_dot_velocity)};
          Scalar<DataType> result{};
          hydro::stress_trace(
              make_not_null(&result), rest_mass_density, specific_enthalpy,
              pressure, spatial_velocity_squared, lorentz_factor,
              magnetic_field_dot_velocity, comoving_magnetic_field_squared);
          // Independently evaluate the fluid kinetic term from v^2, and use
          // the equality of electromagnetic energy density and stress trace.
          // This avoids both W^2 - 1 and the implementation's comoving-field
          // expression. W and v^2 must not be sampled independently.
          const DataType expected =
              3. * get(pressure) +
              get(rest_mass_density) * get(specific_enthalpy) *
                  velocity_squared / (1. - velocity_squared) +
              0.5 * magnetic_field_squared * (1. + velocity_squared) -
              0.5 * get(magnetic_field_dot_velocity) *
                  get(magnetic_field_dot_velocity);
          INFO("stress trace = " << std::scientific << get(result)
                                 << "; reference = " << expected);
          CHECK_ITERABLE_CUSTOM_APPROX(get(result), expected, relative_approx);
        }
      }
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.Hydro.StressTrace",
                  "[Unit][Hydro]") {
  const std::array velocities_squared{0.,    1.e-20, 1.e-16, 1.e-12,
                                      1.e-8, 0.25,   0.75,   0.99};
  for (const double velocity_squared : velocities_squared) {
    test_stress_trace(velocity_squared);
  }
  // Heterogeneous lanes also exercise vector evaluation near W == 1.
  test_stress_trace(
      DataVector{0., 1.e-20, 1.e-16, 1.e-12, 1.e-8, 0.25, 0.75, 0.99});
}
