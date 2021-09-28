// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/KastaunEtAl.hpp"

#include <cmath>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveRecoveryData.hpp"
#include "NumericalAlgorithms/RootFinding/TOMS748.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace grmhd::ValenciaDivClean::PrimitiveRecoverySchemes {

namespace {

// Equation (26)
double compute_x(const double mu, const double b_squared) {
  return 1.0 / (1.0 + mu * b_squared);
}

// Equation (38)
double compute_r_bar_squared(const double mu, const double x,
                             const double r_squared,
                             const double r_dot_b_squared) {
  return x * (r_squared * x + mu * (1.0 + x) * r_dot_b_squared);
}

// Equations (33) and (32)
double compute_v_0_squared(const double r_squared, const double h_0) {
  const double z_0_squared = r_squared / square(h_0);
  static constexpr double velocity_squared_upper_bound =
      1.0 - 4.0 * std::numeric_limits<double>::epsilon();
  return std::min(z_0_squared / (1.0 + z_0_squared),
                  velocity_squared_upper_bound);
}

struct Primitives {
  const double rest_mass_density;
  const double lorentz_factor;
  const double pressure;
  const double specific_internal_energy;
  const double q_bar;
  const double r_bar_squared;
};

// Function to tighten master function bracket in corner case discussed in
// Appendix A when Equation (41) produces a rest mass density outside the
// valid range of the EOS
class CornerCaseFunction {
 public:
  CornerCaseFunction(const double w_target, const double r_squared,
                     const double b_squared, const double r_dot_b_squared)
      : v_squared_target_(1.0 - 1.0 / square(w_target)),
        r_squared_(r_squared),
        b_squared_(b_squared),
        r_dot_b_squared_(r_dot_b_squared) {}

  double operator()(const double mu) const {
    const double x = compute_x(mu, b_squared_);
    const double r_bar_squared =
        compute_r_bar_squared(mu, x, r_squared_, r_dot_b_squared_);
    // v = mu*r_bar, see text after Equation (31)
    // target is v satisfying W(mu) = D/ rho_{min/max}
    return square(mu) * r_bar_squared - v_squared_target_;
  }

 private:
  const double v_squared_target_;
  const double r_squared_;
  const double b_squared_;
  const double r_dot_b_squared_;
};

// Function whose root is upper bracket of master function, see Sec. II.F
class AuxiliaryFunction {
 public:
  AuxiliaryFunction(const double h_0, const double r_squared,
                    const double b_squared, const double r_dot_b_squared)
      : h_0_(h_0),
        r_squared_(r_squared),
        b_squared_(b_squared),
        r_dot_b_squared_(r_dot_b_squared) {}

  double operator()(double mu) const {
    const double x = compute_x(mu, b_squared_);
    const double r_bar_squared =
        compute_r_bar_squared(mu, x, r_squared_, r_dot_b_squared_);
    // Equation (49)
    return mu * sqrt(square(h_0_) + r_bar_squared) - 1.0;
  }

 private:
  const double h_0_;
  const double r_squared_;
  const double b_squared_;
  const double r_dot_b_squared_;
};

// Master function, see Equation (44) in Sec. II.E
template <size_t ThermodynamicDim>
class FunctionOfMu {
 public:
  FunctionOfMu(const double total_energy_density,
               const double momentum_density_squared,
               const double momentum_density_dot_magnetic_field,
               const double magnetic_field_squared,
               const double rest_mass_density_times_lorentz_factor,
               const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
                   equation_of_state)
      : q_(total_energy_density / rest_mass_density_times_lorentz_factor - 1.0),
        r_squared_(momentum_density_squared /
                   square(rest_mass_density_times_lorentz_factor)),
        b_squared_(magnetic_field_squared /
                   rest_mass_density_times_lorentz_factor),
        r_dot_b_squared_(square(momentum_density_dot_magnetic_field) /
                         cube(rest_mass_density_times_lorentz_factor)),
        rest_mass_density_times_lorentz_factor_(
            rest_mass_density_times_lorentz_factor),
        equation_of_state_(equation_of_state),
        h_0_(equation_of_state_.specific_enthalpy_lower_bound()),
        v_0_squared_(compute_v_0_squared(r_squared_, h_0_)) {}

  std::pair<double, double> root_bracket(
      double rest_mass_density_times_lorentz_factor, double absolute_tolerance,
      double relative_tolerance, size_t max_iterations) const;

  Primitives primitives(double mu) const;

  double operator()(const double mu) const;

 private:
  const double q_;
  const double r_squared_;
  const double b_squared_;
  const double r_dot_b_squared_;
  const double rest_mass_density_times_lorentz_factor_;
  const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
      equation_of_state_;
  const double h_0_;
  const double v_0_squared_;
};

template <size_t ThermodynamicDim>
std::pair<double, double> FunctionOfMu<ThermodynamicDim>::root_bracket(
    const double rest_mass_density_times_lorentz_factor,
    const double absolute_tolerance, const double relative_tolerance,
    const size_t max_iterations) const {
  // see text between Equations (49) and (50) and after Equation (54)
  double lower_bound = 0.0;
  // We use `1 / (h_0_ + numeric_limits<double>::min())` to avoid division by
  // zero in a way that avoids conditionals.
  double upper_bound = 1.0 / (h_0_ + std::numeric_limits<double>::min());
  if (r_squared_ < square(h_0_)) {
    // need to solve auxiliary function to determine mu_+ which will
    // be the upper bound for the master function bracket
    const auto auxiliary_function =
        AuxiliaryFunction{h_0_, r_squared_, b_squared_, r_dot_b_squared_};
    upper_bound =
        // NOLINTNEXTLINE(clang-analyzer-core)
        RootFinder::toms748(auxiliary_function, lower_bound, upper_bound,
                            absolute_tolerance, relative_tolerance,
                            max_iterations);
  }

  // Determine if the corner case discussed in Appendix A occurs where the
  // mass density is outside the valid range of the EOS
  const double rho_min = equation_of_state_.rest_mass_density_lower_bound();
  const double rho_max = equation_of_state_.rest_mass_density_upper_bound();

  // If this is triggering, the most likely cause is that the density cutoff
  // for atmosphere is smaller than the minimum density of the EOS, i.e. this
  // point should have been flagged as atmosphere
  if (rest_mass_density_times_lorentz_factor < rho_min) {
    throw std::runtime_error("Density too small for EOS");
  }

  const double mu_b = upper_bound;
  const double x = compute_x(mu_b, b_squared_);
  const double r_bar_squared =
      compute_r_bar_squared(mu_b, x, r_squared_, r_dot_b_squared_);
  // Equation (40)
  const double v_hat_squared =
      std::min(square(mu_b) * r_bar_squared, v_0_squared_);
  const double w_hat = 1.0 / sqrt(1.0 - v_hat_squared);

  // If this is being triggered, the only possible recourse would be
  // to limit the rest mass density to the maximum value allowed by the
  // EOS.  This seems questionable, so it is treated as an inversion failure.
  // The exception should be caught by the try-catch in apply() which will
  // return std::nullopt
  if (rest_mass_density_times_lorentz_factor / w_hat > rho_max) {
    throw std::runtime_error("Density too big for EOS");
  }

  if (rest_mass_density_times_lorentz_factor / w_hat < rho_min) {
    // adjust lower bound
    const auto corner_case_function =
        CornerCaseFunction{rest_mass_density_times_lorentz_factor / rho_max,
                           r_squared_, b_squared_, r_dot_b_squared_};
    lower_bound = std::max(
        lower_bound, RootFinder::toms748(corner_case_function, lower_bound,
                                         upper_bound, absolute_tolerance,
                                         relative_tolerance, max_iterations));
  }

  if (rho_max < rest_mass_density_times_lorentz_factor) {
    // adjust upper bound
    const auto corner_case_function =
        CornerCaseFunction{rest_mass_density_times_lorentz_factor / rho_min,
                           r_squared_, b_squared_, r_dot_b_squared_};
    upper_bound = std::min(
        upper_bound, RootFinder::toms748(corner_case_function, lower_bound,
                                         upper_bound, absolute_tolerance,
                                         relative_tolerance, max_iterations));
  }

  return {lower_bound, upper_bound};
}

template <size_t ThermodynamicDim>
Primitives FunctionOfMu<ThermodynamicDim>::primitives(const double mu) const {
  // Equation (26)
  const double x = compute_x(mu, b_squared_);
  // Equations(38)
  const double r_bar_squared =
      compute_r_bar_squared(mu, x, r_squared_, r_dot_b_squared_);
  // Equation (40)
  const double v_hat_squared =
      std::min(square(mu) * r_bar_squared, v_0_squared_);
  const double w_hat = 1.0 / sqrt(1.0 - v_hat_squared);
  // Equation (41) with bounds from Equation (5)
  const double rho_hat =
      std::clamp(rest_mass_density_times_lorentz_factor_ / w_hat,
                 equation_of_state_.rest_mass_density_lower_bound(),
                 equation_of_state_.rest_mass_density_upper_bound());
  // Equations (39) and (25)
  const double q_bar =
      q_ - 0.5 * b_squared_ -
      0.5 * square(mu * x) * (r_squared_ * b_squared_ - r_dot_b_squared_);
  // Equation (42) with bounds from Equation (6)
  const double epsilon_hat = std::clamp(
      w_hat * (q_bar - mu * r_bar_squared) +
          v_hat_squared * square(w_hat) / (1.0 + w_hat),
      equation_of_state_.specific_internal_energy_lower_bound(rho_hat),
      equation_of_state_.specific_internal_energy_upper_bound(rho_hat));
  // Pressure from EOS
  double p_hat = std::numeric_limits<double>::signaling_NaN();
  if constexpr (ThermodynamicDim == 1) {
    p_hat =
        get(equation_of_state_.pressure_from_density(Scalar<double>(rho_hat)));
  } else if constexpr (ThermodynamicDim == 2) {
    p_hat = get(equation_of_state_.pressure_from_density_and_energy(
        Scalar<double>(rho_hat), Scalar<double>(epsilon_hat)));
  }
  return Primitives{rho_hat, w_hat, p_hat, epsilon_hat, q_bar, r_bar_squared};
}

template <size_t ThermodynamicDim>
double FunctionOfMu<ThermodynamicDim>::operator()(const double mu) const {
  const auto [rho_hat, w_hat, p_hat, epsilon_hat, q_bar, r_bar_squared] =
      primitives(mu);
  // Equation (43)
  const double a_hat = p_hat / (rho_hat * (1.0 + epsilon_hat));
  const double h_hat = (1.0 + epsilon_hat) * (1.0 + a_hat);
  // Equations (46) - (48)
  const double nu_hat = std::max(
      h_hat / w_hat, (1.0 + a_hat) * (1.0 + q_bar - mu * r_bar_squared));
  // Equations (44) - (45)
  return mu - 1.0 / (nu_hat + mu * r_bar_squared);
}
}  // namespace

template <size_t ThermodynamicDim>
std::optional<PrimitiveRecoveryData> KastaunEtAl::apply(
    const double /*initial_guess_pressure*/, const double total_energy_density,
    const double momentum_density_squared,
    const double momentum_density_dot_magnetic_field,
    const double magnetic_field_squared,
    const double rest_mass_density_times_lorentz_factor,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) {
  // Master function see Equation (44)
  const auto f_of_mu =
      FunctionOfMu<ThermodynamicDim>{total_energy_density,
                                     momentum_density_squared,
                                     momentum_density_dot_magnetic_field,
                                     magnetic_field_squared,
                                     rest_mass_density_times_lorentz_factor,
                                     equation_of_state};

  // mu is 1 / (h W) see Equation (26)
  double one_over_specific_enthalpy_times_lorentz_factor =
      std::numeric_limits<double>::signaling_NaN();
  try {
    // Bracket for master function, see Sec. II.F
    const auto [lower_bound, upper_bound] = f_of_mu.root_bracket(
        rest_mass_density_times_lorentz_factor, absolute_tolerance_,
        relative_tolerance_, max_iterations_);

    // Try to recover primitves
    one_over_specific_enthalpy_times_lorentz_factor =
        // NOLINTNEXTLINE(clang-analyzer-core)
        RootFinder::toms748(f_of_mu, lower_bound, upper_bound,
                            absolute_tolerance_, relative_tolerance_,
                            max_iterations_);
  } catch (std::exception& exception) {
    return std::nullopt;
  }

  const auto [rest_mass_density, lorentz_factor, pressure,
              specific_internal_energy, q_bar, r_bar_squared] =
      f_of_mu.primitives(one_over_specific_enthalpy_times_lorentz_factor);

  (void)(specific_internal_energy);
  (void)(q_bar);
  (void)(r_bar_squared);

  return PrimitiveRecoveryData{
      rest_mass_density, lorentz_factor, pressure,
      rest_mass_density_times_lorentz_factor /
          one_over_specific_enthalpy_times_lorentz_factor};
}
}  // namespace grmhd::ValenciaDivClean::PrimitiveRecoverySchemes

#define THERMODIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATION(_, data)                                                \
  template std::optional<grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::  \
                             PrimitiveRecoveryData>                           \
  grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::KastaunEtAl::apply<      \
      THERMODIM(data)>(                                                       \
      const double initial_guess_pressure, const double total_energy_density, \
      const double momentum_density_squared,                                  \
      const double momentum_density_dot_magnetic_field,                       \
      const double magnetic_field_squared,                                    \
      const double rest_mass_density_times_lorentz_factor,                    \
      const EquationsOfState::EquationOfState<true, THERMODIM(data)>&         \
          equation_of_state);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2))

#undef INSTANTIATION
#undef THERMODIM
