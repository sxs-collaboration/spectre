// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/Tov.hpp"

// Need Boost MultiArray because it is used internally by ODEINT
#include "DataStructures/BoostMultiArray.hpp"

#include <algorithm>
#include <array>
#include <boost/numeric/odeint.hpp>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <ostream>
#include <pup.h>
#include <string>
#include <type_traits>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/RootFinding/QuadraticEquation.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/SoundSpeedSquared.hpp"
#include "PointwiseFunctions/Hydro/SpecificEnthalpy.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace RelativisticEuler::Solutions {
std::ostream& operator<<(std::ostream& os, const TovCoordinates coords) {
  switch (coords) {
    case TovCoordinates::Schwarzschild:
      return os << "Schwarzschild";
    case TovCoordinates::Isotropic:
      return os << "Isotropic";
    default:
      ERROR("Unknown TovCoordinates");
  }
}
}  // namespace RelativisticEuler::Solutions

template <>
RelativisticEuler::Solutions::TovCoordinates
Options::create_from_yaml<RelativisticEuler::Solutions::TovCoordinates>::create<
    void>(const Options::Option& options) {
  const auto type_read = options.parse_as<std::string>();
  if ("Schwarzschild" == type_read) {
    return RelativisticEuler::Solutions::TovCoordinates::Schwarzschild;
  } else if ("Isotropic" == type_read) {
    return RelativisticEuler::Solutions::TovCoordinates::Isotropic;
  }
  PARSE_ERROR(
      options.context(),
      "Failed to convert '"
          << type_read
          << "' to RelativisticEuler::Solutions::TovCoordinates. Must be "
             "'Schwarzschild' or 'Isotropic'.");
}

namespace RelativisticEuler::Solutions {
namespace {

// In Schwarzschild coords we integrate u=r^2 and v=m/r (2 vars), and in
// isotropic coords we also integrate w=ln(psi) (3 vars).
template <TovCoordinates CoordSystem>
using TovVars =
    std::conditional_t<CoordSystem == TovCoordinates::Schwarzschild,
                       std::array<double, 2>, std::array<double, 3>>;

/*This function calculates the first two coefficients
of the central Taylor series expansion of the TOV equations.
The derivative of energy density with respect to pressure is
calculated using the sound speed squared, which is calculated
from the equation of state, so this function calculates the
expansion coefficients exactly.
*/
std::pair<std::vector<double>, std::vector<double>> expansion_coeffs(
    const double central_log_enthalpy,
    const EquationsOfState::EquationOfState<true, 1>& equation_of_state) {
  const double specific_enthalpy = std::exp(central_log_enthalpy);
  const double central_rest_mass_density =
      specific_enthalpy == 1.0
          ? 0.0
          : get(equation_of_state.rest_mass_density_from_enthalpy(
                Scalar<double>{std::exp(central_log_enthalpy)}));

  const double central_pressure =
      specific_enthalpy == 1.0
          ? 0.0
          : get(equation_of_state.pressure_from_density(
                Scalar<double>{central_rest_mass_density}));

  const double central_energy_density =
      std::exp(central_log_enthalpy) * central_rest_mass_density -
      central_pressure;

  const double central_specific_internal_energy =
      get(equation_of_state.specific_internal_energy_from_density(
          Scalar<double>{central_rest_mass_density}));

  const double sound_speed_squared = get(hydro::sound_speed_squared(
      Scalar<double>{central_rest_mass_density},
      Scalar<double>{central_specific_internal_energy},
      Scalar<double>{std::exp(central_log_enthalpy)}, equation_of_state));

  const double dedp = 1.0 / sound_speed_squared;

  const double e_1 = -dedp * (central_energy_density + central_pressure);

  const double u_1 =
      3.0 / (2.0 * M_PI * (central_energy_density + 3.0 * central_pressure));
  const double v_1 = 2.0 * central_energy_density /
                     (central_energy_density + 3.0 * central_pressure);

  const double u_2 =
      (15 * (3 * central_pressure - central_energy_density) - 9 * e_1) /
      (20 * M_PI *
       std::pow((central_energy_density + 3.0 * central_pressure), 2));
  const double v_2 =
      (5 * central_energy_density *
           (3 * central_pressure - central_energy_density) +
       3 * (central_energy_density + 6 * central_pressure) * e_1) /
      (5 * std::pow((central_energy_density + 3.0 * central_pressure), 2));

  return std::pair<std::vector<double>, std::vector<double>>{
      std::vector<double>{u_1, u_2}, std::vector<double>{v_1, v_2}};
}

/*This function estimates the second derivative of the
energy density with respect to pressure as a means of
calculating the third-order terms of the central Taylor
series expansion of the TOV equations. The derivative
estimate uses a 4th order finite differencing method.
*/
double deriv_estimate(
    const double central_log_enthalpy, const double delH,
    const EquationsOfState::EquationOfState<true, 1>& equation_of_state) {
  const double specific_enthalpy = std::exp(central_log_enthalpy);
  const double central_rest_mass_density =  // get rmd at center
      specific_enthalpy == 1.0
          ? 0.0
          : get(equation_of_state.rest_mass_density_from_enthalpy(
                Scalar<double>{std::exp(central_log_enthalpy)}));

  const double central_pressure =
      specific_enthalpy == 1.0  // get presesure at center
          ? 0.0
          : get(equation_of_state.pressure_from_density(
                Scalar<double>{central_rest_mass_density}));

  const double
      central_specific_internal_energy =  // get specific internal energy
      get(equation_of_state.specific_internal_energy_from_density(
          Scalar<double>{central_rest_mass_density}));

  const double central_energy_density =
      specific_enthalpy * central_rest_mass_density - central_pressure;

  const double sound_speed_squared = get(hydro::sound_speed_squared(
      Scalar<double>{central_rest_mass_density},
      Scalar<double>{central_specific_internal_energy},
      Scalar<double>{std::exp(central_log_enthalpy)}, equation_of_state));

  const double rmd_2plus =
      get(equation_of_state.rest_mass_density_from_enthalpy(
          Scalar<double>{std::exp(central_log_enthalpy + 2 * delH)}));
  const double rmd_plus = get(equation_of_state.rest_mass_density_from_enthalpy(
      Scalar<double>{std::exp(central_log_enthalpy + delH)}));
  const double rmd_minus =
      get(equation_of_state.rest_mass_density_from_enthalpy(
          Scalar<double>{std::exp(central_log_enthalpy - delH)}));
  const double rmd_2minus =
      get(equation_of_state.rest_mass_density_from_enthalpy(
          Scalar<double>{std::exp(central_log_enthalpy - 2 * delH)}));

  const double specific_internal_energy_plus =
      get(equation_of_state.specific_internal_energy_from_density(
          Scalar<double>{rmd_plus}));
  const double specific_internal_energy_minus =
      get(equation_of_state.specific_internal_energy_from_density(
          Scalar<double>{rmd_minus}));
  const double specific_internal_energy_2plus =
      get(equation_of_state.specific_internal_energy_from_density(
          Scalar<double>{rmd_2plus}));
  const double specific_internal_energy_2minus =
      get(equation_of_state.specific_internal_energy_from_density(
          Scalar<double>{rmd_2minus}));

  const double sound_speed_squared_2plus = get(hydro::sound_speed_squared(
      Scalar<double>{rmd_2plus}, Scalar<double>{specific_internal_energy_2plus},
      Scalar<double>{std::exp(central_log_enthalpy + 2 * delH)},
      equation_of_state));
  const double sound_speed_squared_plus = get(hydro::sound_speed_squared(
      Scalar<double>{rmd_plus}, Scalar<double>{specific_internal_energy_plus},
      Scalar<double>{std::exp(central_log_enthalpy + delH)},
      equation_of_state));
  const double sound_speed_squared_minus = get(hydro::sound_speed_squared(
      Scalar<double>{rmd_minus}, Scalar<double>{specific_internal_energy_minus},
      Scalar<double>{std::exp(central_log_enthalpy - delH)},
      equation_of_state));
  const double sound_speed_squared_2minus = get(hydro::sound_speed_squared(
      Scalar<double>{rmd_2minus},
      Scalar<double>{specific_internal_energy_2minus},
      Scalar<double>{std::exp(central_log_enthalpy - 2 * delH)},
      equation_of_state));
  const double d2edp2 =
      -(1.0 /
        (std::pow(sound_speed_squared, 2) *
         (central_energy_density + central_pressure)) *
        (sound_speed_squared_2minus - 8 * sound_speed_squared_minus +
         8 * sound_speed_squared_plus - sound_speed_squared_2plus) /
        (12 * delH));
  return d2edp2;
}
/*This function estimates the third-order terms of the central
Taylor series expansion of the TOV equations. This function
utilizes the derivative estimate of the second derivative
of energy density with respect to pressure to calculate the
third-order terms of the expansion. This coefficient is not exact,
but it has been found to be relatively accurate for simple EOS cases
like the Polytropic EOS. The third-order terms are used to estimate
the dynamically calculated threshold for the Taylor series expansion,
which is used to start the integration of the TOV equations at a small
radius away from the center of the star.
*/
std::pair<double, double> third_order_estimate(
    const double central_log_enthalpy, const double delH,
    const EquationsOfState::EquationOfState<true, 1>& equation_of_state) {
  const double specific_enthalpy = std::exp(central_log_enthalpy);
  const double central_rest_mass_density =  // get rmd at center
      specific_enthalpy == 1.0
          ? 0.0
          : get(equation_of_state.rest_mass_density_from_enthalpy(
                Scalar<double>{std::exp(central_log_enthalpy)}));

  const double central_pressure =
      specific_enthalpy == 1.0  // get pressure at center
          ? 0.0
          : get(equation_of_state.pressure_from_density(
                Scalar<double>{central_rest_mass_density}));

  const double central_energy_density =  // get energy density at center
      std::exp(central_log_enthalpy) * central_rest_mass_density -
      central_pressure;

  const double
      central_specific_internal_energy =  // get specific internal energy
      get(equation_of_state.specific_internal_energy_from_density(
          Scalar<double>{central_rest_mass_density}));

  const double sound_speed_squared = get(hydro::sound_speed_squared(
      Scalar<double>{central_rest_mass_density},
      Scalar<double>{central_specific_internal_energy},
      Scalar<double>{std::exp(central_log_enthalpy)}, equation_of_state));
  const double dedp = 1.0 / sound_speed_squared;
  const double d2edp2 =
      deriv_estimate(central_log_enthalpy, delH, equation_of_state);
  const double e_1 = -dedp * (central_energy_density + central_pressure);
  const double e_2 = 0.5 * (central_energy_density + central_pressure) *
                     ((central_energy_density + central_pressure) * d2edp2 +
                      dedp * (1 + dedp));
  const double sum_ec_3pc = central_energy_density + 3.0 * central_pressure;

  const double u_3 = (3.0 * central_pressure - 5.0 * central_energy_density) /
                         (4.0 * M_PI * std::pow(sum_ec_3pc, 2)) -
                     (3.0 * e_2) / (14.0 * M_PI * std::pow(sum_ec_3pc, 2)) +
                     (3.0 * e_1 *
                      (48.0 * e_1 - 95.0 * central_energy_density -
                       765.0 * central_pressure)) /
                         (700.0 * M_PI * std::pow(sum_ec_3pc, 3));

  const double v_3 =
      (central_energy_density *
       (3.0 * central_pressure - 5.0 * central_energy_density)) /
          (3.0 * std::pow(sum_ec_3pc, 2)) +
      (2.0 * (2.0 * central_energy_density + 9.0 * central_pressure) *
       central_energy_density) /
          (7.0 * std::pow(sum_ec_3pc, 2)) -
      (5.0 *
           (46.0 * std::pow(central_energy_density, 2) +
            153.0 * central_energy_density * central_pressure -
            243.0 * std::pow(central_pressure, 2)) *
           e_1 +
       3.0 * (11.0 * central_energy_density + 81.0 * central_pressure) *
           std::pow(e_1, 2)) /
          (175.0 * std::pow(sum_ec_3pc, 3));
  return std::pair<double, double>{u_3, v_3};
}

/*This function calculates a fallback threshold for
the Taylor series expansion of the TOV equations.
Should the dynamically calculated threshold found from
comparing the third-order estimate of the Taylor series expansion
to the lower-order expansion coefficients be invalid,
the fallback threshold is used to start the integration
of the TOV equations at a small radius away from the center
of the star, as a third order expansion should be valid for
at least as large a radius as the second order expansion.
*/
std::pair<double, double> fallback_thresh(
    const std::pair<std::vector<double>, std::vector<double>>&
        expansion_coeffs_result,
    const double eps) {
  const double u_1 = expansion_coeffs_result.first[0];
  const double u_2 = expansion_coeffs_result.first[1];
  const double v_1 = expansion_coeffs_result.second[0];
  const double v_2 = expansion_coeffs_result.second[1];

  const double u_thresh = std::abs(u_1 / (2 * u_2)) * eps;
  const double v_thresh = std::abs(v_1 / (2 * v_2)) * eps;
  return std::pair<double, double>(u_thresh, v_thresh);
}
/*This function calculates the third-order threshold or returns
nullopt if the threshold is invalid.
*/
std::optional<double> find_thresh(
    const std::pair<std::vector<double>, std::vector<double>>&
        expansion_coeffs_result,
    const std::pair<double, double>& third_order_estimate_result,
    const double eps) {
  const double u_1 = expansion_coeffs_result.first[0];
  const double u_2 = expansion_coeffs_result.first[1];
  const double u_3 = third_order_estimate_result.first;
  const double v_1 = expansion_coeffs_result.second[0];
  const double v_2 = expansion_coeffs_result.second[1];
  const double v_3 = third_order_estimate_result.second;

  // Use the quadratic equation solver to find the threshold
  const double u_a = 3 * std::abs(u_3);
  const double u_b = -2 * std::abs(u_2) * eps;
  const double u_c = -std::abs(u_1) * eps;

  std::optional<std::array<double, 2>> u_roots = real_roots(u_a, u_b, u_c);

  const double v_a = 3 * std::abs(v_3);
  const double v_b = -2 * std::abs(v_2) * eps;
  const double v_c = -std::abs(v_1) * eps;

  std::optional<std::array<double, 2>> v_roots = real_roots(v_a, v_b, v_c);

  if (u_roots.has_value() and v_roots.has_value()) {
    const double u_thresh =
        std::min(std::abs(u_roots.value()[0]), std::abs(u_roots.value()[1]));
    const double v_thresh =
        std::min(std::abs(v_roots.value()[0]), std::abs(v_roots.value()[1]));
    return std::min(u_thresh, v_thresh);
  } else {
    return std::nullopt;
  }
}

template <TovCoordinates CoordSystem>
void lindblom_rhs(
    const gsl::not_null<TovVars<CoordSystem>*> dvars,
    const TovVars<CoordSystem>& vars, const double log_enthalpy,
    const double central_log_enthalpy,
    const std::pair<std::vector<double>, std::vector<double>>&
        expansion_coeffs_result,
    const double thresh,
    const EquationsOfState::EquationOfState<true, 1>& equation_of_state) {
  const double& radius_squared = vars[0];  // u = r^2

  const double& mass_over_radius = vars[1];  // v = m / r

  double& d_radius_squared = (*dvars)[0];  // du/dH = dr^2/dH

  double& d_mass_over_radius = (*dvars)[1];  // dv/dH = d(m/r)/dH

  const double specific_enthalpy = std::exp(log_enthalpy);  // h = e^H

  const double rest_mass_density =
      specific_enthalpy == 1.0
          ? 0.0
          : get(equation_of_state.rest_mass_density_from_enthalpy(
                Scalar<double>{specific_enthalpy}));

  const double pressure = specific_enthalpy == 1.0
                              ? 0.0
                              : get(equation_of_state.pressure_from_density(
                                    Scalar<double>{rest_mass_density}));

  const double energy_density =
      specific_enthalpy * rest_mass_density - pressure;

  // At the center of the star: (u,v) = (0,0)
  const double h_diff_log = central_log_enthalpy - log_enthalpy;

  if (UNLIKELY(h_diff_log <= thresh)) {
    const double u1 = expansion_coeffs_result.first[0];
    const double u2 = expansion_coeffs_result.first[1];
    const double v1 = expansion_coeffs_result.second[0];
    const double v2 = expansion_coeffs_result.second[1];
    d_radius_squared = -u1 - 2 * u2 * h_diff_log;
    d_mass_over_radius = -v1 - 2 * v2 * h_diff_log;
    if constexpr (CoordSystem == TovCoordinates::Isotropic) {
      double& d_log_conformal_factor = (*dvars)[2];
      d_log_conformal_factor = -0.25 * d_mass_over_radius;
    }
  } else {
    const double one_minus_two_m_over_r = 1.0 - 2.0 * mass_over_radius;
    const double denominator =
        4.0 * M_PI * radius_squared * pressure + mass_over_radius;
    const double common_factor = one_minus_two_m_over_r / denominator;
    d_radius_squared = -2.0 * radius_squared * common_factor;
    d_mass_over_radius =
        -(4.0 * M_PI * radius_squared * energy_density - mass_over_radius) *
        common_factor;
    if constexpr (CoordSystem == TovCoordinates::Isotropic) {
      double& d_log_conformal_factor = (*dvars)[2];
      d_log_conformal_factor = sqrt(one_minus_two_m_over_r) /
                               (1.0 + sqrt(one_minus_two_m_over_r)) *
                               mass_over_radius / denominator;
    }
  }
}

template <TovCoordinates CoordSystem>
class IntegralObserver {
 public:
  void operator()(const TovVars<CoordSystem>& vars,
                  const double current_log_enthalpy) {
    radius.push_back(std::sqrt(vars[0]));
    mass_over_radius.push_back(vars[1]);
    if constexpr (CoordSystem == TovCoordinates::Isotropic) {
      conformal_factor.push_back(exp(vars[2]));
    }
    log_enthalpy.push_back(current_log_enthalpy);
  }
  std::vector<double> radius;
  std::vector<double> mass_over_radius;
  std::vector<double> conformal_factor;
  std::vector<double> log_enthalpy;
};

}  // namespace

template <TovCoordinates CoordSystem>
void TovSolution::integrate(
    const EquationsOfState::EquationOfState<true, 1>& equation_of_state,
    const double central_mass_density,
    const double log_enthalpy_at_outer_radius, const double absolute_tolerance,
    const double relative_tolerance) {
  constexpr double eps =
      std::numeric_limits<double>::epsilon();  // machine precision
  using Vars = TovVars<CoordSystem>;
  Vars vars{};
  // Initial integration variables at the center of the star
  vars[0] = 0.;  // u = r^2 = 0
  vars[1] = 0.;  // v = m / r = 0
  if constexpr (CoordSystem == TovCoordinates::Isotropic) {
    vars[2] = 0.;  // w = ln(psi) = 0 (rescaled later)
  }
  Vars dvars{};
  const Scalar<double> central_specific_internal_energy =
      equation_of_state.specific_internal_energy_from_density(
          Scalar<double>{central_mass_density});
  const Scalar<double> central_pressure =
      equation_of_state.pressure_from_density(
          Scalar<double>{central_mass_density});
  const double central_log_enthalpy =
      std::log(get(hydro::relativistic_specific_enthalpy(
          Scalar<double>{central_mass_density},
          central_specific_internal_energy, central_pressure)));

  int status = 0;
  char* demangled = abi::__cxa_demangle(typeid(equation_of_state).name(),
                                        nullptr, nullptr, &status);
  free(demangled);
  // check if the error in the third order estimate is of order del_H^4
  // due to the 4th order finite difference derivative used to compute d^2edp^2.
  // If not, we use the fallback threshold instead of the one computed
  // from the quadratic equation roots.
  const double strict_dH = std::pow(eps, 1.0 / 2.0) * central_log_enthalpy;

  const std::vector<double> test_dH = {
      std::pow(eps, 1.0 / 3.0) * central_log_enthalpy,
      std::pow(eps, 1.0 / 4.0) * central_log_enthalpy,
      std::pow(eps, 1.0 / 5.0) * central_log_enthalpy,
      std::pow(eps, 1.0 / 6.0) * central_log_enthalpy};

  const double strict_d2edp2 =
      deriv_estimate(central_log_enthalpy, strict_dH, equation_of_state);

  bool passed_order4_test = true;

  for (const double& dH : test_dH) {
    const double test_d2edp2 =
        deriv_estimate(central_log_enthalpy, dH, equation_of_state);
    if (!equal_within_roundoff(std::abs(strict_d2edp2) * std::pow(dH, 4),
                               std::abs(test_d2edp2) * std::pow(strict_dH, 4.0),
                               1E-2)) {
      // test gets failed
      passed_order4_test = false;
    }
  }
  int depth_used = 0;
  std::pair<std::vector<double>, std::vector<double>> expansion_coeffs_result =
      expansion_coeffs(central_log_enthalpy, equation_of_state);

  std::pair<double, double> third_order_estimate_result = third_order_estimate(
      central_log_enthalpy, std::pow(eps, 1.0 / 5.0) * central_log_enthalpy,
      equation_of_state);

  std::optional<double> thresh =
      find_thresh(expansion_coeffs_result, third_order_estimate_result, eps);

  double fallback_thresh_val =
      std::min(fallback_thresh(expansion_coeffs_result, eps).first,
               fallback_thresh(expansion_coeffs_result, eps).second);

  // if the threshold is not valid, we use the fallback threshold instead.
  // we check that the estimate of the threshold is positive and greater
  // than the fallback threshold, and that the third order estimate
  // is valid. we also check that the threshold converges to a value
  // that is close to the previous estimate of the threshold,
  // and that the threshold is not zero. if any of these conditions
  // are not met, we use the fallback threshold instead.
  bool use_fallback_thresh = false;
  if (!thresh.has_value() or !passed_order4_test or
      fallback_thresh_val > thresh.value() or thresh.value() == 0.0) {
    thresh = fallback_thresh_val;
    use_fallback_thresh = true;
    depth_used = -1;
  } else {
    std::pair<double, double> third_order_test = third_order_estimate(
        central_log_enthalpy, thresh.value(), equation_of_state);

    std::optional<double> thresh_test =
        find_thresh(expansion_coeffs_result, third_order_test, eps);

    if (thresh_test.has_value() and thresh_test.value() > 0.0 and
        thresh_test.value() > fallback_thresh_val) {
      std::pair<double, double> third_order_test2 = third_order_estimate(
          central_log_enthalpy, thresh_test.value(), equation_of_state);
      std::optional<double> thresh_test2 =
          find_thresh(expansion_coeffs_result, third_order_test2, eps);

      if (thresh_test2.has_value() and thresh_test2.value() > 0.0 and
          thresh_test2.value() > fallback_thresh_val) {
        if (std::abs(thresh_test2.value() - thresh_test.value()) >=
                std::abs(thresh_test.value() - thresh.value()) or
            thresh_test2.value() < fallback_thresh_val) {
          thresh = fallback_thresh_val;
          use_fallback_thresh = true;
          depth_used = -1;
        } else {
          depth_used = 2;
          thresh = thresh_test2.value();
        }
      } else {
        depth_used = 1;
        thresh = thresh_test.value();
      }
    } else {
      thresh = thresh.value();
    }
  }

  lindblom_rhs<CoordSystem>(
      make_not_null(&dvars), vars, central_log_enthalpy, central_log_enthalpy,
      expansion_coeffs_result, thresh.value(),
      equation_of_state);  // this is the INITIAL step, so we use
                           // central_log_enthalpy for both args
  double initial_step =
      -std::min(std::abs(1.0 / dvars[0]), std::abs(1.0 / dvars[1]));
  if constexpr (CoordSystem == TovCoordinates::Isotropic) {
    initial_step = -std::min(std::abs(initial_step), std::abs(1.0 / dvars[2]));
  }
  using StateDopri5 = boost::numeric::odeint::runge_kutta_dopri5<Vars>;
  boost::numeric::odeint::dense_output_runge_kutta<
      boost::numeric::odeint::controlled_runge_kutta<StateDopri5>>
      dopri5 = make_dense_output(absolute_tolerance, relative_tolerance,
                                 StateDopri5{});
  IntegralObserver<CoordSystem> observer{};
  boost::numeric::odeint::integrate_adaptive(
      dopri5,
      [&equation_of_state, central_log_enthalpy, &expansion_coeffs_result,
       thresh](const Vars& local_vars, Vars& local_dvars,
               const double local_enthalpy) {
        return lindblom_rhs<CoordSystem>(
            &local_dvars, local_vars, local_enthalpy, central_log_enthalpy,
            expansion_coeffs_result, thresh.value(),
            equation_of_state);  // passed central_log_enthalpy to the RHS
                                 // function for use in the near-center
                                 // expansion
      },
      vars, central_log_enthalpy, log_enthalpy_at_outer_radius, initial_step,
      std::ref(observer));
  outer_radius_ = observer.radius.back();
  const double total_mass_over_radius = observer.mass_over_radius.back();
  total_mass_ = total_mass_over_radius * outer_radius_;
  injection_energy_ = sqrt(1. - 2. * total_mass_ / outer_radius_);

  if constexpr (CoordSystem == TovCoordinates::Isotropic) {
    // Transform outer radius to isotropic
    const double outer_areal_radius = outer_radius_;
    outer_radius_ = 0.5 * (outer_areal_radius - total_mass_ +
                           sqrt(square(outer_areal_radius) -
                                2. * total_mass_ * outer_areal_radius));

    // Match conformal factor to exterior solution
    const double outer_conformal_factor =
        1.0 + 0.5 * total_mass_ / outer_radius_;
    const double matching_constant =
        outer_conformal_factor / observer.conformal_factor.back();
    const size_t num_points = observer.radius.size();
    for (size_t i = 0; i < num_points; ++i) {
      observer.conformal_factor[i] *= matching_constant;
      // Transform observed radius to isotropic, so we use the isotropic radius
      // for all interpolations below
      observer.radius[i] /= square(observer.conformal_factor[i]);
      // The interpolation is not safe otherwise
    }
    observer.radius.back() = outer_radius_;
    conformal_factor_interpolant_ =
        intrp::CubicSpline(observer.radius, observer.conformal_factor);
  }

  mass_over_radius_interpolant_ =
      intrp::CubicSpline(observer.radius, observer.mass_over_radius);
  // log_enthalpy(radius) is almost linear so an interpolant of order 3
  // maximizes precision
  log_enthalpy_interpolant_ =
      intrp::CubicSpline(observer.radius, observer.log_enthalpy);
}

TovSolution::TovSolution(
    const EquationsOfState::EquationOfState<true, 1>& equation_of_state,
    const double central_mass_density, const TovCoordinates coordinate_system,
    const double log_enthalpy_at_outer_radius, const double absolute_tolerance,
    const double relative_tolerance)
    : coordinate_system_(coordinate_system) {
  if (coordinate_system_ == TovCoordinates::Schwarzschild) {
    integrate<TovCoordinates::Schwarzschild>(
        equation_of_state, central_mass_density, log_enthalpy_at_outer_radius,
        absolute_tolerance, relative_tolerance);
  } else {
    integrate<TovCoordinates::Isotropic>(
        equation_of_state, central_mass_density, log_enthalpy_at_outer_radius,
        absolute_tolerance, relative_tolerance);
  }
}

template <typename DataType>
DataType TovSolution::mass_over_radius(const DataType& r) const {
  // Possible optimization: Support DataVector in intrp::BarycentricRational
  auto result = make_with_value<DataType>(r, 0.);
  for (size_t i = 0; i < get_size(r); ++i) {
    ASSERT(
        get_element(r, i) >= 0.0 and get_element(r, i) <= outer_radius_,
        "Invalid radius: " << r << " not in [0.0, " << outer_radius_ << "]\n");
    get_element(result, i) = mass_over_radius_interpolant_(get_element(r, i));
  }
  return result;
}

template <typename DataType>
DataType TovSolution::log_specific_enthalpy(const DataType& r) const {
  // Possible optimization: Support DataVector in intrp::BarycentricRational
  auto result = make_with_value<DataType>(r, 0.);
  for (size_t i = 0; i < get_size(r); ++i) {
    ASSERT(
        get_element(r, i) >= 0.0 and get_element(r, i) <= outer_radius_,
        "Invalid radius: " << r << " not in [0.0, " << outer_radius_ << "]\n");
    get_element(result, i) = log_enthalpy_interpolant_(get_element(r, i));
  }
  return result;
}

template <typename DataType>
DataType TovSolution::conformal_factor(const DataType& r) const {
  ASSERT(coordinate_system_ == TovCoordinates::Isotropic,
         "The conformal factor is computed only for isotropic coordinates.");
  // Possible optimization: Support DataVector in intrp::BarycentricRational
  auto result = make_with_value<DataType>(r, 0.);
  for (size_t i = 0; i < get_size(r); ++i) {
    ASSERT(
        get_element(r, i) >= 0.0 and get_element(r, i) <= outer_radius_,
        "Invalid radius: " << r << " not in [0.0, " << outer_radius_ << "]\n");
    get_element(result, i) = conformal_factor_interpolant_(get_element(r, i));
  }
  return result;
}

void TovSolution::pup(PUP::er& p) {  // NOLINT
  p | coordinate_system_;
  p | outer_radius_;
  p | total_mass_;
  p | injection_energy_;
  p | mass_over_radius_interpolant_;
  p | log_enthalpy_interpolant_;
  if (coordinate_system_ == TovCoordinates::Isotropic) {
    p | conformal_factor_interpolant_;
  }
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                \
  template DTYPE(data) TovSolution::mass_over_radius(const DTYPE(data) & r) \
      const;                                                                \
  template DTYPE(data)                                                      \
      TovSolution::log_specific_enthalpy(const DTYPE(data) & r) const;      \
  template DTYPE(data) TovSolution::conformal_factor(const DTYPE(data) & r) \
      const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef DTYPE

}  // namespace RelativisticEuler::Solutions
