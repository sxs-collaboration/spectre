// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Systems/GrMhd/ValenciaDivClean/PalenzuelaEtAl.hpp"

#include <cmath>
#include <exception>
#include <limits>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveRecoveryData.hpp"
#include "NumericalAlgorithms/RootFinding/TOMS748.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "Utilities/ConstantExpressions.hpp"

namespace grmhd::ValenciaDivClean::PrimitiveRecoverySchemes {

namespace PalenzuelaEtAl_detail {

// note q,r,s,t,x are defined in the documentation
template <typename EosType>
class FunctionOfX {
 public:
  FunctionOfX(const double tau, const double momentum_density_squared,
              const double momentum_density_dot_magnetic_field,
              const double magnetic_field_squared,
              const double rest_mass_density_times_lorentz_factor,
              const double electron_fraction, const EosType& equation_of_state)
      : q_(tau / rest_mass_density_times_lorentz_factor),
        r_(momentum_density_squared /
           square(rest_mass_density_times_lorentz_factor)),
        s_(magnetic_field_squared / rest_mass_density_times_lorentz_factor),
        t_squared_(square(momentum_density_dot_magnetic_field) /
                   cube(rest_mass_density_times_lorentz_factor)),
        rest_mass_density_times_lorentz_factor_(
            rest_mass_density_times_lorentz_factor),
        electron_fraction_(electron_fraction),
        equation_of_state_(equation_of_state) {}

  double lorentz_factor(const double x) const {
    static constexpr double v_maximum = 1.0 - 1.e-12;
    // Clamp v^2 to physical values.  This is needed because the bounds on
    // x used for the root solve do not guarantee a physical velocity.  Some
    // work would be needed to investigate whether better bounds could guarantee
    // a physical velocity.
    const double v_squared = std::clamp(
        (square(x) * r_ + (2 * x + s_) * t_squared_) / square(x * (x + s_)),
        0.0, square(v_maximum));
    return 1.0 / sqrt(1.0 - v_squared);
  }

  double specific_internal_energy(const double x,
                                  const double lorentz_factor) const {
    return lorentz_factor - 1.0 +
           x * (1.0 - square(lorentz_factor)) / lorentz_factor +
           lorentz_factor * (q_ - s_ + 0.5 * t_squared_ / square(x) +
                             0.5 * s_ / square(lorentz_factor));
  }

  double operator()(const double x) const {
    const double current_lorentz_factor = lorentz_factor(x);
    const double current_rest_mass_density =
        rest_mass_density_times_lorentz_factor_ / current_lorentz_factor;
    const double current_specific_internal_energy =
        specific_internal_energy(x, current_lorentz_factor);
    double current_pressure = std::numeric_limits<double>::signaling_NaN();
    if constexpr (EosType::thermodynamic_dim == 1) {
      current_pressure = get(equation_of_state_.pressure_from_density(
          Scalar<double>(current_rest_mass_density)));
    } else if constexpr (EosType::thermodynamic_dim == 2) {
      current_pressure =
          get(equation_of_state_.pressure_from_density_and_energy(
              Scalar<double>(current_rest_mass_density),
              Scalar<double>(current_specific_internal_energy)));
    } else if constexpr (EosType::thermodynamic_dim == 3) {
      current_pressure =
          get(equation_of_state_.pressure_from_density_and_energy(
              Scalar<double>(current_rest_mass_density),
              Scalar<double>(current_specific_internal_energy),
              Scalar<double>(electron_fraction_)));
    }

    return x - (1.0 + current_specific_internal_energy +
                current_pressure / current_rest_mass_density) *
                   current_lorentz_factor;
  }

 private:
  const double q_;
  const double r_;
  const double s_;
  const double t_squared_;
  const double rest_mass_density_times_lorentz_factor_;
  const double electron_fraction_;
  const EosType& equation_of_state_;
};
}  // namespace PalenzuelaEtAl_detail

template <bool EnforcePhysicality, typename EosType>
std::optional<PrimitiveRecoveryData> PalenzuelaEtAl::apply(
    const double /*initial_guess_pressure*/, const double tau,
    const double momentum_density_squared,
    const double momentum_density_dot_magnetic_field,
    const double magnetic_field_squared,
    const double rest_mass_density_times_lorentz_factor,
    const double electron_fraction, const EosType& equation_of_state,
    const grmhd::ValenciaDivClean::PrimitiveFromConservativeOptions&
    /*primitive_from_conservative_options*/) {
  const double lower_bound =
      (tau - magnetic_field_squared) / rest_mass_density_times_lorentz_factor +
      1.0;
  const double upper_bound = (2.0 * tau - magnetic_field_squared) /
                                 rest_mass_density_times_lorentz_factor +
                             2.0;
  const auto f_of_x = PalenzuelaEtAl_detail::FunctionOfX<EosType>{
      tau,
      momentum_density_squared,
      momentum_density_dot_magnetic_field,
      magnetic_field_squared,
      rest_mass_density_times_lorentz_factor,
      electron_fraction,
      equation_of_state};
  double specific_enthalpy_times_lorentz_factor =
      std::numeric_limits<double>::signaling_NaN();
  try {
    specific_enthalpy_times_lorentz_factor =
        // NOLINTNEXTLINE(clang-analyzer-core)
        RootFinder::toms748(f_of_x, lower_bound, upper_bound,
                            absolute_tolerance_, relative_tolerance_,
                            max_iterations_);
  } catch (std::exception& exception) {
    return std::nullopt;
  }
  const double lorentz_factor =
      f_of_x.lorentz_factor(specific_enthalpy_times_lorentz_factor);
  const double rest_mass_density =
      rest_mass_density_times_lorentz_factor / lorentz_factor;
  double pressure = std::numeric_limits<double>::signaling_NaN();
  double specific_internal_energy = f_of_x.specific_internal_energy(
      specific_enthalpy_times_lorentz_factor, lorentz_factor);
  if constexpr (EosType::thermodynamic_dim == 1) {
    pressure = get(equation_of_state.pressure_from_density(
        Scalar<double>(rest_mass_density)));
    specific_internal_energy =
        get(equation_of_state.specific_internal_energy_from_density(
            Scalar<double>(rest_mass_density)));
  } else if constexpr (EosType::thermodynamic_dim == 2) {
    pressure = get(equation_of_state.pressure_from_density_and_energy(
        Scalar<double>(rest_mass_density),
        Scalar<double>(specific_internal_energy)));
  } else if constexpr (EosType::thermodynamic_dim == 3) {
    pressure = get(equation_of_state.pressure_from_density_and_energy(
        Scalar<double>(rest_mass_density),
        Scalar<double>(specific_internal_energy),
        Scalar<double>(electron_fraction)));
  }

  return PrimitiveRecoveryData{rest_mass_density,
                               lorentz_factor,
                               pressure,
                               specific_internal_energy,
                               specific_enthalpy_times_lorentz_factor *
                                   rest_mass_density_times_lorentz_factor,
                               electron_fraction};
}
}  // namespace grmhd::ValenciaDivClean::PrimitiveRecoverySchemes
