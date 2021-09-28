// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Tov.hpp"

// Need Boost MultiArray because it is used internally by ODEINT
#include "DataStructures/BoostMultiArray.hpp"  // IWYU pragma: keep

#include <algorithm>
#include <array>
#include <boost/numeric/odeint.hpp>  // IWYU pragma: keep
#include <cmath>
#include <cstddef>
#include <functional>
#include <ostream>
#include <pup.h>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

// IWYU pragma: no_include <boost/numeric/odeint/integrate/integrate_adaptive.hpp>
// IWYU pragma: no_include <boost/numeric/odeint/stepper/controlled_runge_kutta.hpp>
// IWYU pragma: no_include <boost/numeric/odeint/stepper/dense_output_runge_kutta.hpp>
// IWYU pragma: no_include <boost/numeric/odeint/stepper/generation/make_dense_output.hpp>
// IWYU pragma: no_include <boost/numeric/odeint/stepper/runge_kutta_dopri5.hpp>
// IWYU pragma: no_include <complex>

// IWYU pragma: no_forward_declare boost::numeric::odeint::controlled_runge_kutta
// IWYU pragma: no_forward_declare EquationsOfState::EquationOfState
// IWYU pragma: no_forward_declare Tensor

namespace {

void lindblom_rhs(
    const gsl::not_null<std::array<double, 2>*> dvars,
    const std::array<double, 2>& vars, const double log_enthalpy,
    const EquationsOfState::EquationOfState<true, 1>& equation_of_state) {
  const double& radius_squared = vars[0];
  const double& mass_over_radius = vars[1];
  double& d_radius_squared = (*dvars)[0];
  double& d_mass_over_radius = (*dvars)[1];
  const double specific_enthalpy = std::exp(log_enthalpy);
  const double rest_mass_density =
      get(equation_of_state.rest_mass_density_from_enthalpy(
          Scalar<double>{specific_enthalpy}));
  const double pressure = get(equation_of_state.pressure_from_density(
      Scalar<double>{rest_mass_density}));
  const double energy_density =
      specific_enthalpy * rest_mass_density - pressure;

  // At the center of the star: (u,v) = (0,0)
  if (UNLIKELY((radius_squared == 0.0) and (mass_over_radius == 0.0))) {
    d_radius_squared = -3.0 / (2.0 * M_PI * (energy_density + 3.0 * pressure));
    d_mass_over_radius =
        -2.0 * energy_density / (energy_density + 3.0 * pressure);
  } else {
    const double common_factor =
        (1.0 - 2.0 * mass_over_radius) /
        (4.0 * M_PI * radius_squared * pressure + mass_over_radius);
    d_radius_squared = -2.0 * radius_squared * common_factor;
    d_mass_over_radius =
        -(4.0 * M_PI * radius_squared * energy_density - mass_over_radius) *
        common_factor;
  }
}

class Observer {
 public:
  void operator()(const std::array<double, 2>& vars,
                  const double current_log_enthalpy) {
    radius.push_back(std::sqrt(vars[0]));
    mass_over_radius.push_back(vars[1]);
    log_enthalpy.push_back(current_log_enthalpy);
  }
  std::vector<double> radius;
  std::vector<double> mass_over_radius;
  std::vector<double> log_enthalpy;
};

template <typename DataType>
RelativisticEuler::Solutions::TovStar<
    gr::Solutions::TovSolution>::RadialVariables<DataType>
interior_solution(
    const EquationsOfState::EquationOfState<true, 1>& equation_of_state,
    const DataType& radius, const DataType& mass_over_radius,
    const DataType& log_specific_enthalpy,
    const double log_lapse_at_outer_radius) {
  RelativisticEuler::Solutions::TovStar<
      gr::Solutions::TovSolution>::RadialVariables<DataType>
      result(radius);
  result.specific_enthalpy = Scalar<DataType>{exp(log_specific_enthalpy)};
  result.rest_mass_density = equation_of_state.rest_mass_density_from_enthalpy(
      result.specific_enthalpy);
  result.pressure =
      equation_of_state.pressure_from_density(result.rest_mass_density);
  result.specific_internal_energy =
      equation_of_state.specific_internal_energy_from_density(
          result.rest_mass_density);
  // From the TOV equation, write radial derivative of the pressure as
  //   - (rho(1.0+epsilon)+P)*((m/r)+4*pi r^2 P) / (r - 2 * r (m/r))
  // where rho is the rest-mass density and epsilon the specific internal
  // energy.
  for (size_t i = 0; i < get_size(radius); ++i) {
    if (get_element(radius, i) > 0.0) {
      get_element(result.dr_pressure, i) =
          -(get_element(get(result.rest_mass_density), i) *
                (1.0 + get_element(get(result.specific_internal_energy), i)) +
            get_element(get(result.pressure), i)) *
          (get_element(mass_over_radius, i) +
           4.0 * M_PI * get_element(get(result.pressure), i) *
               square(get_element(radius, i))) /
          (get_element(radius, i) *
           (1.0 - 2.0 * get_element(mass_over_radius, i)));
    } else {
      get_element(result.dr_pressure, i) = 0.0;
    }
  }
  result.metric_time_potential =
      log_lapse_at_outer_radius - log_specific_enthalpy;
  result.dr_metric_time_potential =
      (mass_over_radius / radius + 4.0 * M_PI * get(result.pressure) * radius) /
      (1.0 - 2.0 * mass_over_radius);
  result.metric_radial_potential = -0.5 * log(1.0 - 2.0 * mass_over_radius);
  result.dr_metric_radial_potential =
      (4.0 * M_PI * radius *
           (get(result.specific_enthalpy) * get(result.rest_mass_density) -
            get(result.pressure)) -
       mass_over_radius / radius) /
      (1.0 - 2.0 * mass_over_radius);
  result.metric_angular_potential = make_with_value<DataType>(radius, 0.0);
  result.dr_metric_angular_potential = make_with_value<DataType>(radius, 0.0);
  return result;
}

template <typename DataType>
RelativisticEuler::Solutions::TovStar<
    gr::Solutions::TovSolution>::RadialVariables<DataType>
vacuum_solution(const DataType& radius, const double total_mass) {
  RelativisticEuler::Solutions::TovStar<
      gr::Solutions::TovSolution>::RadialVariables<DataType>
      result(radius);
  result.specific_enthalpy = make_with_value<Scalar<DataType>>(radius, 1.0);
  result.rest_mass_density = make_with_value<Scalar<DataType>>(radius, 0.0);
  result.pressure = make_with_value<Scalar<DataType>>(radius, 0.0);
  result.specific_internal_energy =
      make_with_value<Scalar<DataType>>(radius, 0.0);
  result.dr_pressure = make_with_value<DataType>(radius, 0.0);
  const DataType one_minus_two_m_over_r = 1.0 - 2.0 * total_mass / radius;
  result.metric_time_potential = 0.5 * log(one_minus_two_m_over_r);
  result.dr_metric_time_potential =
      total_mass / square(radius) / one_minus_two_m_over_r;
  result.metric_radial_potential = -result.metric_time_potential;
  result.dr_metric_radial_potential = -result.dr_metric_time_potential;
  result.metric_angular_potential = make_with_value<DataType>(radius, 0.0);
  result.dr_metric_angular_potential = make_with_value<DataType>(radius, 0.0);
  return result;
}

}  // namespace

namespace gr::Solutions {

TovSolution::TovSolution(
    const EquationsOfState::EquationOfState<true, 1>& equation_of_state,
    const double central_mass_density,
    const double log_enthalpy_at_outer_radius, const double absolute_tolerance,
    const double relative_tolerance) {
  std::array<double, 2> u_and_v = {{0.0, 0.0}};
  std::array<double, 2> dudh_and_dvdh{};
  const double central_log_enthalpy =
      std::log(get(equation_of_state.specific_enthalpy_from_density(
          Scalar<double>{central_mass_density})));
  lindblom_rhs(&dudh_and_dvdh, u_and_v, central_log_enthalpy,
               equation_of_state);
  const double initial_step = -std::min(std::abs(1.0 / dudh_and_dvdh[0]),
                                        std::abs(1.0 / dudh_and_dvdh[1]));
  using StateDopri5 =
      boost::numeric::odeint::runge_kutta_dopri5<std::array<double, 2>>;
  boost::numeric::odeint::dense_output_runge_kutta<
      boost::numeric::odeint::controlled_runge_kutta<StateDopri5>>
      dopri5 = make_dense_output(absolute_tolerance, relative_tolerance,
                                 StateDopri5{});
  Observer observer{};
  boost::numeric::odeint::integrate_adaptive(
      dopri5,
      [&equation_of_state](const std::array<double, 2>& lindblom_u_and_v,
                           std::array<double, 2>& lindblom_dudh_and_dvdh,
                           const double lindblom_enthalpy) {
        return lindblom_rhs(&lindblom_dudh_and_dvdh, lindblom_u_and_v,
                            lindblom_enthalpy, equation_of_state);
      },
      u_and_v, central_log_enthalpy, log_enthalpy_at_outer_radius, initial_step,
      std::ref(observer));
  outer_radius_ = observer.radius.back();
  const double total_mass_over_radius = observer.mass_over_radius.back();
  total_mass_ = total_mass_over_radius * outer_radius_;
  log_lapse_at_outer_radius_ = 0.5 * log(1.0 - 2.0 * total_mass_over_radius);
  mass_over_radius_interpolant_ =
      intrp::BarycentricRational(observer.radius, observer.mass_over_radius, 5);
  // log_enthalpy(radius) is almost linear so an interpolant of order 3
  // maximizes precision
  log_enthalpy_interpolant_ =
      intrp::BarycentricRational(observer.radius, observer.log_enthalpy, 3);
}

double TovSolution::outer_radius() const { return outer_radius_; }

double TovSolution::mass_over_radius(const double r) const {
  ASSERT(r >= 0.0 and r <= outer_radius_,
         "Invalid radius: " << r << " not in [0.0, " << outer_radius_ << "]\n");
  return mass_over_radius_interpolant_(r);
}

double TovSolution::mass(const double r) const {
  return mass_over_radius(r) * r;
}

double TovSolution::log_specific_enthalpy(const double r) const {
  ASSERT(r >= 0.0 and r <= outer_radius_,
         "Invalid radius: " << r << " not in [0.0, " << outer_radius_ << "]\n");
  return log_enthalpy_interpolant_(r);
}

template <>
RelativisticEuler::Solutions::TovStar<TovSolution>::RadialVariables<double>
TovSolution::radial_variables(
    const EquationsOfState::EquationOfState<true, 1>& equation_of_state,
    const tnsr::I<double, 3>& x) const {
  // add small number to avoid FPEs at origin
  const double radius = get(magnitude(x)) + 1.e-30 * outer_radius_;
  if (radius >= outer_radius_) {
    return vacuum_solution(radius, total_mass_);
  }
  return interior_solution(equation_of_state, radius, mass_over_radius(radius),
                           log_specific_enthalpy(radius),
                           log_lapse_at_outer_radius_);
}

template <>
RelativisticEuler::Solutions::TovStar<TovSolution>::RadialVariables<DataVector>
TovSolution::radial_variables(
    const EquationsOfState::EquationOfState<true, 1>& equation_of_state,
    const tnsr::I<DataVector, 3>& x) const {
  // add small number to avoid FPEs at origin
  const DataVector radius = get(magnitude(x)) + 1.e-30 * outer_radius_;
  if (min(radius) >= outer_radius_) {
    return vacuum_solution(radius, total_mass_);
  }
  if (max(radius) <= outer_radius_) {
    DataVector mass_over_radius_data(radius.size());
    DataVector log_of_specific_enthalpy(radius.size());
    for (size_t i = 0; i < get_size(radius); i++) {
      const double r = get_element(radius, i);
      get_element(mass_over_radius_data, i) = mass_over_radius(r);
      get_element(log_of_specific_enthalpy, i) = log_specific_enthalpy(r);
    }
    return interior_solution(equation_of_state, radius, mass_over_radius_data,
                             log_of_specific_enthalpy,
                             log_lapse_at_outer_radius_);
  }
  RelativisticEuler::Solutions::TovStar<TovSolution>::RadialVariables<
      DataVector>
      result(radius);
  for (size_t i = 0; i < radius.size(); i++) {
    const double r = radius[i];
    auto radial_vars_at_r =
        (r <= outer_radius_
             ? interior_solution(equation_of_state, r, mass_over_radius(r),
                                 log_specific_enthalpy(r),
                                 log_lapse_at_outer_radius_)
             : vacuum_solution(r, total_mass_));
    get(result.rest_mass_density)[i] = get(radial_vars_at_r.rest_mass_density);
    get(result.pressure)[i] = get(radial_vars_at_r.pressure);
    get(result.specific_internal_energy)[i] =
        get(radial_vars_at_r.specific_internal_energy);
    get(result.specific_enthalpy)[i] = get(radial_vars_at_r.specific_enthalpy);
    result.dr_pressure[i] = radial_vars_at_r.dr_pressure;
    result.metric_time_potential[i] = radial_vars_at_r.metric_time_potential;
    result.dr_metric_time_potential[i] =
        radial_vars_at_r.dr_metric_time_potential;
    result.metric_radial_potential[i] =
        radial_vars_at_r.metric_radial_potential;
    result.dr_metric_radial_potential[i] =
        radial_vars_at_r.dr_metric_radial_potential;
  }
  result.metric_angular_potential = make_with_value<DataVector>(radius, 0.0);
  result.dr_metric_angular_potential = make_with_value<DataVector>(radius, 0.0);
  return result;
}

void TovSolution::pup(PUP::er& p) {  // NOLINT
  p | outer_radius_;
  p | total_mass_;
  p | log_lapse_at_outer_radius_;
  p | mass_over_radius_interpolant_;
  p | log_enthalpy_interpolant_;
}

}  // namespace gr::Solutions
