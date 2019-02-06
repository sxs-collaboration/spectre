// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/NewmanHamlin.hpp"

#include <algorithm>
#include <array>
#include <boost/none.hpp>
#include <cmath>
#include <limits>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveRecoveryData.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Overloader.hpp"

// IWYU pragma: no_forward_declare EquationsOfState::EquationOfState

/// \cond
namespace grmhd {
namespace ValenciaDivClean {
namespace PrimitiveRecoverySchemes {

template <size_t ThermodynamicDim>
boost::optional<PrimitiveRecoveryData> NewmanHamlin::apply(
    const double initial_guess_for_pressure, const double total_energy_density,
    const double momentum_density_squared,
    const double momentum_density_dot_magnetic_field,
    const double magnetic_field_squared,
    const double rest_mass_density_times_lorentz_factor,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) noexcept {
  // constant in cubic equation  f(eps) = eps^3 - a eps^2 + d
  // whose root is being found at each point in the iteration below
  const double d_in_cubic = [
    momentum_density_squared, magnetic_field_squared,
    momentum_density_dot_magnetic_field
  ]() noexcept {
    const double local_d_in_cubic =
        0.5 * (momentum_density_squared * magnetic_field_squared -
               square(momentum_density_dot_magnetic_field));
    if (UNLIKELY(-1e-12 * square(momentum_density_dot_magnetic_field) >
                 local_d_in_cubic)) {
      return local_d_in_cubic;  // will fail returning boost::none
    }
    return std::max(0.0, local_d_in_cubic);
  }
  ();
  if (UNLIKELY(0.0 > d_in_cubic)) {
    return boost::none;
  }

  // bound needed so cubic equation has a positive root
  const double minimum_pressure =
      std::max(0.0, cbrt(6.75 * d_in_cubic) - total_energy_density -
                        0.5 * magnetic_field_squared);
  double current_pressure =
      std::max(minimum_pressure, initial_guess_for_pressure);
  double previous_pressure{std::numeric_limits<double>::signaling_NaN()};
  size_t iteration_step{0};
  std::array<double, 3> aitken_pressure{
      {current_pressure, std::numeric_limits<double>::signaling_NaN(),
       std::numeric_limits<double>::signaling_NaN()}};
  size_t valid_entries_in_aitken_pressure = 1;
  bool converged = false;

  while (true) {  // will break when relative pressure change is <
                  // relative_tolernance_
    if (UNLIKELY(max_iterations_ == iteration_step and not converged)) {
      return boost::none;
    }

    ++iteration_step;
    previous_pressure = current_pressure;
    // enforces NH Eq.(5.9): d <= (4/27) a^3 so cubic has positive root
    current_pressure = std::max(current_pressure, minimum_pressure);
    const double a_in_cubic =
        total_energy_density + current_pressure + 0.5 * magnetic_field_squared;

    if (UNLIKELY(a_in_cubic < 0.0)) {
      return boost::none;
    }

    // NH Eq. (5.10): d = (4/27) a^3 cos^2(phi)
    const double phi = acos(sqrt(6.75 * d_in_cubic / cube(a_in_cubic)));
    // NH Eq. (5.11) with l=1 is desired positive root
    const double root_of_cubic =
        (a_in_cubic / 3.0) * (1.0 - 2.0 * cos((2.0 / 3.0) * (M_PI + phi)));
    // NH Eq. (5.5) with their script L being rho_h_w_squared
    // where rho is rest_mass_density, h is specific_enthalpy,
    // and w is the lorentz factor
    const double rho_h_w_squared = root_of_cubic - magnetic_field_squared;

    if (UNLIKELY(rho_h_w_squared <= 0.0)) {
      return boost::none;
    }

    // NH Eq. (5.2) with (5.5) substituted in denominator
    const double v_squared =
        (momentum_density_squared * square(rho_h_w_squared) +
         square(momentum_density_dot_magnetic_field) *
             (magnetic_field_squared + 2.0 * rho_h_w_squared)) /
        square(rho_h_w_squared * root_of_cubic);

    // If this fails, there was code in the Bitbucket version that adjusted
    // the pressure to get the maximum allowed velocity in atmosphere.
    // Instead, we could return boost::none and try the next inversion method.
    if (UNLIKELY(v_squared < 0.0 or v_squared >= 1.0)) {
      return boost::none;
    }

    const double current_lorentz_factor = sqrt(1.0 / (1.0 - v_squared));
    const double current_rest_mass_density =
        rest_mass_density_times_lorentz_factor / current_lorentz_factor;

    if (converged) {
      return PrimitiveRecoveryData{current_rest_mass_density,
                                   current_lorentz_factor, current_pressure,
                                   rho_h_w_squared};
    }

    const double current_specific_enthalpy = [
      rho_h_w_squared, current_rest_mass_density, current_lorentz_factor
    ]() noexcept {
      const double specific_enthalpy =
          rho_h_w_squared /
          (current_rest_mass_density * square(current_lorentz_factor));
      if (UNLIKELY(1.0 - 1.0e-12 > specific_enthalpy)) {
        return specific_enthalpy;  // will fail returning boost::none
      }
      return std::max(1.0, specific_enthalpy);
    }
    ();
    if (UNLIKELY(1.0 > current_specific_enthalpy)) {
      return boost::none;
    }

    current_pressure = get(make_overloader(
        [&current_rest_mass_density](
            const EquationsOfState::EquationOfState<true, 1>&
                the_equation_of_state) noexcept {
          return the_equation_of_state.pressure_from_density(
              Scalar<double>(current_rest_mass_density));
        },
        [&current_rest_mass_density, &current_specific_enthalpy ](
            const EquationsOfState::EquationOfState<true, 2>&
                the_equation_of_state) noexcept {
          return the_equation_of_state.pressure_from_density_and_enthalpy(
              Scalar<double>(current_rest_mass_density),
              Scalar<double>(current_specific_enthalpy));
        })(equation_of_state));

    gsl::at(aitken_pressure, valid_entries_in_aitken_pressure++) =
        current_pressure;
    if (3 == valid_entries_in_aitken_pressure) {
      const double aitken_residual = (aitken_pressure[2] - aitken_pressure[1]) /
                                     (aitken_pressure[1] - aitken_pressure[0]);
      if (0.0 <= aitken_residual and aitken_residual < 1.0) {
        previous_pressure = current_pressure;
        current_pressure =
            aitken_pressure[1] +
            (aitken_pressure[2] - aitken_pressure[1]) / (1.0 - aitken_residual);
        aitken_pressure = {{current_pressure,
                            std::numeric_limits<double>::signaling_NaN(),
                            std::numeric_limits<double>::signaling_NaN()}};
        valid_entries_in_aitken_pressure = 1;
      } else {
        // Aitken extrapolation failed, retain latest 2 values for next attempt
        aitken_pressure[0] = aitken_pressure[1];
        aitken_pressure[1] = aitken_pressure[2];
        valid_entries_in_aitken_pressure = 2;
      }
    }
    // note primitives are recomputed above before being returned
    converged = fabs(current_pressure - previous_pressure) <=
                relative_tolerance_ * (current_pressure + previous_pressure);
  }  // while loop
}
}  // namespace PrimitiveRecoverySchemes
}  // namespace ValenciaDivClean
}  // namespace grmhd

#define THERMODIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATION(_, data)                                                 \
  template boost::optional<grmhd::ValenciaDivClean::PrimitiveRecoverySchemes:: \
                               PrimitiveRecoveryData>                          \
  grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::NewmanHamlin::apply<      \
      THERMODIM(data)>(                                                        \
      const double initial_guess_pressure, const double total_energy_density,  \
      const double momentum_density_squared,                                   \
      const double momentum_density_dot_magnetic_field,                        \
      const double magnetic_field_squared,                                     \
      const double rest_mass_density_times_lorentz_factor,                     \
      const EquationsOfState::EquationOfState<true, THERMODIM(data)>&          \
          equation_of_state) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2))

#undef INSTANTIATION
#undef THERMODIM
/// \endcond
