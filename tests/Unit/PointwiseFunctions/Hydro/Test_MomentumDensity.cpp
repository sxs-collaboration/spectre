// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <limits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Hydro/StressEnergy.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {
template <typename DataType>
void test_exact_parallel_field(const DataType& speed) {
  using std::sqrt;
  // Each component of v is 0 or +/- 1/2, so v^2 is 0 or 3/4 and W is
  // exactly 1 or 2. With B = (b,b,b), every magnetic intermediate is an
  // exact binary fraction for the chosen b. The Poynting momentum cancels
  // exactly, leaving only the small fluid momentum this regression tests.
  auto velocity = make_with_value<tnsr::I<DataType, 3>>(speed, 0.);
  for (size_t d = 0; d < 3; ++d) {
    velocity.get(d) = speed;
  }
  const Scalar<DataType> lorentz_factor{1. / sqrt(1. - 3. * speed * speed)};
  const auto relative_approx =
      Approx::custom()
          .epsilon(128. * std::numeric_limits<double>::epsilon())
          .scale(0.);
  for (const double density : {0., 1.e-20, 1., 2.}) {
    for (const double pressure_over_density : {0., 0.125}) {
      const auto rest_mass_density =
          make_with_value<Scalar<DataType>>(speed, density);
      const auto specific_enthalpy = make_with_value<Scalar<DataType>>(
          speed, 1. + 3. * pressure_over_density);
      for (const double field_strength : {0., 0.5, std::ldexp(1., 30)}) {
        CAPTURE(speed, density, pressure_over_density, field_strength);
        const auto magnetic_field =
            make_with_value<tnsr::I<DataType, 3>>(speed, field_strength);
        const Scalar<DataType> magnetic_field_dot_velocity{3. * field_strength *
                                                           speed};
        const auto comoving_magnetic_field_squared =
            make_with_value<Scalar<DataType>>(
                speed, 3. * field_strength * field_strength);
        tnsr::I<DataType, 3> result{};
        hydro::momentum_density(make_not_null(&result), rest_mass_density,
                                specific_enthalpy, velocity, lorentz_factor,
                                magnetic_field, magnetic_field_dot_velocity,
                                comoving_magnetic_field_squared);
        // For each nonzero speed, W^2 v^i = +/- 2 exactly. At rest the
        // momentum vanishes. No magnetic-size absolute tolerance is used.
        const DataType expected = 4. * density * get(specific_enthalpy) * speed;
        for (size_t d = 0; d < 3; ++d) {
          CHECK_ITERABLE_CUSTOM_APPROX(result.get(d), expected,
                                       relative_approx);
        }
      }
    }
  }
}

template <typename DataType>
void test_momentum_controls(const DataType& speed) {
  using std::sqrt;
  const Scalar<DataType> lorentz_factor{1. / sqrt(1. - speed * speed)};
  for (size_t velocity_axis = 0; velocity_axis < 3; ++velocity_axis) {
    auto spatial_velocity = make_with_value<tnsr::I<DataType, 3>>(speed, 0.);
    spatial_velocity.get(velocity_axis) = speed;
    for (const double density : {0., 1.e-20, 2.}) {
      for (const double pressure_over_density : {0., 0.125}) {
        // Gamma = 3/2 gamma-law states, including their zero-density limit.
        const auto rest_mass_density =
            make_with_value<Scalar<DataType>>(speed, density);
        const auto specific_enthalpy = make_with_value<Scalar<DataType>>(
            speed, 1. + 3. * pressure_over_density);
        for (const double field_strength : {0., 0.5, 1.e10}) {
          // These general-orientation controls allow rounding on the
          // electromagnetic scale. They do not establish retention of a
          // much smaller fluid term; the exact parallel cases above do.
          const auto control_approx =
              Approx::custom()
                  .epsilon(128. * std::numeric_limits<double>::epsilon())
                  .scale(field_strength * field_strength);
          for (const bool parallel_field : {false, true}) {
            CAPTURE(speed, velocity_axis, density, pressure_over_density,
                    field_strength, parallel_field);
            auto magnetic_field =
                make_with_value<tnsr::I<DataType, 3>>(speed, 0.);
            magnetic_field.get(parallel_field ? velocity_axis
                                              : (velocity_axis + 1) % 3) =
                make_with_value<DataType>(speed, field_strength);
            const Scalar<DataType> magnetic_field_dot_velocity{
                parallel_field ? field_strength * speed
                               : make_with_value<DataType>(speed, 0.)};
            const Scalar<DataType> comoving_magnetic_field_squared{
                field_strength * field_strength /
                    (get(lorentz_factor) * get(lorentz_factor)) +
                get(magnetic_field_dot_velocity) *
                    get(magnetic_field_dot_velocity)};
            tnsr::I<DataType, 3> result{};
            hydro::momentum_density(
                make_not_null(&result), rest_mass_density, specific_enthalpy,
                spatial_velocity, lorentz_factor, magnetic_field,
                magnetic_field_dot_velocity, comoving_magnetic_field_squared);
            // The Poynting momentum is zero for a parallel field, and B^2 v
            // for a perpendicular field. Neither reference subtracts two
            // large magnetic terms. Evaluate the fluid contribution from v.
            const DataType expected =
                density * get(specific_enthalpy) * speed /
                    (1. - speed * speed) +
                (parallel_field ? 0. : field_strength * field_strength) * speed;
            for (size_t d = 0; d < 3; ++d) {
              CHECK_ITERABLE_CUSTOM_APPROX(
                  result.get(d),
                  d == velocity_axis ? expected
                                     : make_with_value<DataType>(speed, 0.),
                  control_approx);
            }
          }
        }
      }
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.Hydro.MomentumDensity",
                  "[Unit][Hydro]") {
  for (const double speed : {0., 0.5, -0.5}) {
    test_exact_parallel_field(speed);
  }
  test_exact_parallel_field(DataVector{0., 0.5, -0.5});
  for (const double speed : {0., 0.6, -0.6}) {
    test_momentum_controls(speed);
  }
  test_momentum_controls(DataVector{0., 0.6, -0.6});
}
