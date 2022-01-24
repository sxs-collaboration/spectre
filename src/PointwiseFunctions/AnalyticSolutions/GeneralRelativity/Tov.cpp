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
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
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
  injection_energy_ = sqrt(1. - 2. * total_mass_ / outer_radius_);
  mass_over_radius_interpolant_ =
      intrp::BarycentricRational(observer.radius, observer.mass_over_radius, 5);
  // log_enthalpy(radius) is almost linear so an interpolant of order 3
  // maximizes precision
  log_enthalpy_interpolant_ =
      intrp::BarycentricRational(observer.radius, observer.log_enthalpy, 3);
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

void TovSolution::pup(PUP::er& p) {  // NOLINT
  p | outer_radius_;
  p | total_mass_;
  p | injection_energy_;
  p | mass_over_radius_interpolant_;
  p | log_enthalpy_interpolant_;
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                \
  template DTYPE(data) TovSolution::mass_over_radius(const DTYPE(data) & r) \
      const;                                                                \
  template DTYPE(data)                                                      \
      TovSolution::log_specific_enthalpy(const DTYPE(data) & r) const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef DTYPE

}  // namespace gr::Solutions
