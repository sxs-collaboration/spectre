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
#include <string>
#include <type_traits>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace gr::Solutions {
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
}  // namespace gr::Solutions

template <>
gr::Solutions::TovCoordinates
Options::create_from_yaml<gr::Solutions::TovCoordinates>::create<void>(
    const Options::Option& options) {
  const auto type_read = options.parse_as<std::string>();
  if ("Schwarzschild" == type_read) {
    return gr::Solutions::TovCoordinates::Schwarzschild;
  } else if ("Isotropic" == type_read) {
    return gr::Solutions::TovCoordinates::Isotropic;
  }
  PARSE_ERROR(options.context(),
              "Failed to convert '"
                  << type_read
                  << "' to gr::Solutions::TovCoordinates. Must be "
                     "'Schwarzschild' or 'Isotropic'.");
}

namespace gr::Solutions {
namespace {

// In Schwarzschild coords we integrate u=r^2 and v=m/r (2 vars), and in
// isotropic coords we also integrate w=ln(psi) (3 vars).
template <TovCoordinates CoordSystem>
using TovVars =
    std::conditional_t<CoordSystem == TovCoordinates::Schwarzschild,
                       std::array<double, 2>, std::array<double, 3>>;

template <TovCoordinates CoordSystem>
void lindblom_rhs(
    const gsl::not_null<TovVars<CoordSystem>*> dvars,
    const TovVars<CoordSystem>& vars, const double log_enthalpy,
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
  using Vars = TovVars<CoordSystem>;
  Vars vars{};
  // Initial integration variables at the center of the star
  vars[0] = 0.;  // u = r^2 = 0
  vars[1] = 0.;  // v = m / r = 0
  if constexpr (CoordSystem == TovCoordinates::Isotropic) {
    vars[2] = 0.;  // w = ln(psi) = 0 (rescaled later)
  }
  Vars dvars{};
  const double central_log_enthalpy =
      std::log(get(equation_of_state.specific_enthalpy_from_density(
          Scalar<double>{central_mass_density})));
  lindblom_rhs<CoordSystem>(make_not_null(&dvars), vars, central_log_enthalpy,
                            equation_of_state);
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
      [&equation_of_state](const Vars& local_vars, Vars& local_dvars,
                           const double local_enthalpy) {
        return lindblom_rhs<CoordSystem>(&local_dvars, local_vars,
                                         local_enthalpy, equation_of_state);
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
    }
    conformal_factor_interpolant_ = intrp::BarycentricRational(
        observer.radius, observer.conformal_factor, 5);
  }

  mass_over_radius_interpolant_ =
      intrp::BarycentricRational(observer.radius, observer.mass_over_radius, 5);
  // log_enthalpy(radius) is almost linear so an interpolant of order 3
  // maximizes precision
  log_enthalpy_interpolant_ =
      intrp::BarycentricRational(observer.radius, observer.log_enthalpy, 3);
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
  p | conformal_factor_interpolant_;
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

}  // namespace gr::Solutions
