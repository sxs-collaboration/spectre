// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/KastaunEtAlHydro.hpp"

#include <cmath>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveRecoveryData.hpp"
#include "NumericalAlgorithms/RootFinding/TOMS748.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace grmhd::ValenciaDivClean::PrimitiveRecoverySchemes {

namespace {

struct Primitives {
  const double rest_mass_density;
  const double lorentz_factor;
  const double pressure;
  const double specific_internal_energy;
};

template <size_t ThermodynamicDim, bool EnforcePhysicality>
class FunctionOfZ {
 public:
  FunctionOfZ(const double tau, const double momentum_density_squared,
              const double /* momentum_density_dot_magnetic_field */,
              const double /* magnetic_field_squared */,
              const double rest_mass_density_times_lorentz_factor,
              const double electron_fraction,
              const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
                  equation_of_state)
      : q_(tau / rest_mass_density_times_lorentz_factor),
        r_squared_(momentum_density_squared /
                   square(rest_mass_density_times_lorentz_factor)),
        r_(std::sqrt(r_squared_)),
        rest_mass_density_times_lorentz_factor_(
            rest_mass_density_times_lorentz_factor),
        electron_fraction_(electron_fraction),
        equation_of_state_(equation_of_state) {
    // Internal consistency check
    //
    const double rho_min =
        rest_mass_density_times_lorentz_factor_ / lorentz_max_;
    const double eps_min =
        equation_of_state_.specific_internal_energy_lower_bound(rho_min);

    if constexpr (EnforcePhysicality) {
      q_ = std::max(q_, eps_min);
      r_ = std::min(r_, kmax_ * (q_ + 1.));
    } else {
      if ((q_ < eps_min) or (r_ > kmax_ * (q_ + 1.))) {
        state_is_unphysical_ = true;
      }
    }
  }

  std::pair<double, double> root_bracket();

  Primitives primitives(double z) const;

  double operator()(double z) const;

  bool has_no_root() { return state_is_unphysical_; };

 private:
  double q_;
  double r_squared_;
  double r_;
  bool state_is_unphysical_ = false;
  const double rest_mass_density_times_lorentz_factor_;
  const double electron_fraction_;
  const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
      equation_of_state_;
  const double lorentz_max_ = 1000.;
  const double v_0_squared_ = 1. - 1. / (lorentz_max_ * lorentz_max_);
  const double kmax_ = 2. * std::sqrt(v_0_squared_) / (1. + v_0_squared_);
};

template <size_t ThermodynamicDim, bool EnforcePhysicality>
std::pair<double, double>
FunctionOfZ<ThermodynamicDim, EnforcePhysicality>::root_bracket() {
  auto k = r_ / (q_ + 1.);

  const double rho_min = equation_of_state_.rest_mass_density_lower_bound();

  // If this is triggering, the most likely cause is that the density cutoff
  // for atmosphere is smaller than the minizm density of the EOS, i.e. this
  // point should have been flagged as atmosphere
  if (rest_mass_density_times_lorentz_factor_ < rho_min) {
    throw std::runtime_error("Density too small for EOS");
  }

  // Compute bounds (C23)
  double lower_bound = 0.5 * k / std::sqrt(std::abs(1. - 0.25 * k * k));
  // Ensure that upper_bound does not become degenerate when k ~ 0
  // Empirically, an offset of 1.e-8 has worked well.
  double upper_bound = 1.e-8 + k / std::sqrt(std::abs(1. - k * k));

  return {lower_bound, upper_bound};
}

template <size_t ThermodynamicDim, bool EnforcePhysicality>
Primitives FunctionOfZ<ThermodynamicDim, EnforcePhysicality>::primitives(
    double z) const {
  // Compute Lorentz factor, note that z = lorentz * v
  const double w_hat = std::sqrt(1. + z * z);

  // Compute rest mass density, D/w_hat
  const double rho_hat =
      std::clamp(rest_mass_density_times_lorentz_factor_ / w_hat,
                 equation_of_state_.rest_mass_density_lower_bound(),
                 equation_of_state_.rest_mass_density_upper_bound());

  // Equation (C14) and (C16)
  const double epsilon_hat = std::clamp(
      w_hat * q_ - z * (r_ - z / (1. + w_hat)),
      equation_of_state_.specific_internal_energy_lower_bound(rho_hat),
      equation_of_state_.specific_internal_energy_upper_bound(rho_hat));
  // Pressure from EOS
  double p_hat = std::numeric_limits<double>::signaling_NaN();
  if constexpr (ThermodynamicDim == 1) {
    // Note: we do not reset epsilon to satisfy the EOS because in that case
    // we are not guaranteed to be able to find a solution. That's because
    // the conserved-variable solution is inconsistent with the restrictive
    // 1d-EOS. Instead, we recover the primitives and then reset the
    // specific internal energy and specific enthalpy using the EOS.
    p_hat =
        get(equation_of_state_.pressure_from_density(Scalar<double>(rho_hat)));
  } else if constexpr (ThermodynamicDim == 2) {
    p_hat = get(equation_of_state_.pressure_from_density_and_energy(
        Scalar<double>(rho_hat), Scalar<double>(epsilon_hat)));
  } else if constexpr (ThermodynamicDim == 3) {
    ERROR("3d EOS not implemented");
  }
  return Primitives{rho_hat, w_hat, p_hat, epsilon_hat};
}

template <size_t ThermodynamicDim, bool EnforcePhysicality>
double FunctionOfZ<ThermodynamicDim, EnforcePhysicality>::operator()(
    double z) const {
  const auto[rho_hat, w_hat, p_hat, epsilon_hat] = primitives(z);
  // Equation (C5)
  const double a_hat = p_hat / (rho_hat * (1.0 + epsilon_hat));
  const double h_hat = (1.0 + epsilon_hat) * (1.0 + a_hat);

  // Equations (C22)
  return z - r_ / h_hat;
}
}  // namespace

template <bool EnforcePhysicality, size_t ThermodynamicDim>
std::optional<PrimitiveRecoveryData> KastaunEtAlHydro::apply(
    const double /*initial_guess_pressure*/, const double tau,
    const double momentum_density_squared,
    const double momentum_density_dot_magnetic_field,
    const double magnetic_field_squared,
    const double rest_mass_density_times_lorentz_factor,
    const double electron_fraction,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) {
  // Master function see Equation (44)
  auto f_of_z = FunctionOfZ<ThermodynamicDim, EnforcePhysicality>{
      tau,
      momentum_density_squared,
      momentum_density_dot_magnetic_field,
      magnetic_field_squared,
      rest_mass_density_times_lorentz_factor,
      electron_fraction,
      equation_of_state};

  if (f_of_z.has_no_root()) {
    return std::nullopt;
  };

  // z is W * v  (Lorentz factor * velocity)
  double z = std::numeric_limits<double>::signaling_NaN();
  try {
    // Bracket for master function
    const auto[lower_bound, upper_bound] = f_of_z.root_bracket();

    // Try to recover primitves
    z =
        // NOLINTNEXTLINE(clang-analyzer-core)
        RootFinder::toms748(f_of_z, lower_bound, upper_bound,
                            absolute_tolerance_, relative_tolerance_,
                            max_iterations_);
  } catch (std::exception& exception) {
    return std::nullopt;
  }

  const auto[rest_mass_density, lorentz_factor, pressure,
             specific_internal_energy] = f_of_z.primitives(z);

  return PrimitiveRecoveryData{
      rest_mass_density,
      lorentz_factor,
      pressure,
      specific_internal_energy,
      (rest_mass_density * (1. + specific_internal_energy) + pressure) *
          (1. + z * z),
      electron_fraction};
}
}  // namespace grmhd::ValenciaDivClean::PrimitiveRecoverySchemes

#define THERMODIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define PHYSICALITY(data) BOOST_PP_TUPLE_ELEM(1, data)
#define INSTANTIATION(_, data)                                                \
  template std::optional<grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::  \
                             PrimitiveRecoveryData>                           \
  grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::KastaunEtAlHydro::apply< \
      PHYSICALITY(data), THERMODIM(data)>(                                    \
      const double initial_guess_pressure, const double tau,                  \
      const double momentum_density_squared,                                  \
      const double momentum_density_dot_magnetic_field,                       \
      const double magnetic_field_squared,                                    \
      const double rest_mass_density_times_lorentz_factor,                    \
      const double electron_fraction,                                         \
      const EquationsOfState::EquationOfState<true, THERMODIM(data)>&         \
          equation_of_state);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2), (true, false))

#undef INSTANTIATION
#undef THERMODIM
#undef PHYSICALITY
