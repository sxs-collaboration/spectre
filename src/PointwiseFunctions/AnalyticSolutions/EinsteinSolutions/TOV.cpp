// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <boost/array.hpp>
#include <boost/math/interpolators/barycentric_rational.hpp>
#include <boost/math/tools/roots.hpp>
#include <boost/numeric/odeint.hpp>
#include <boost/range/adaptors.hpp>
#include <functional>

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"        // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"     // IWYU pragma: keep
#include "PointwiseFunctions/AnalyticSolutions/EinsteinSolutions/TOV.hpp"
#include "PointwiseFunctions/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/EquationsOfState/PolytropicFluid.hpp"

// Uses Lindblom's method of integrating the TOV equations from
// "Phase Transitions and the Mass-Radius Curves of Relativistic Stars"
// L. Lindblom, Phys.Rev. D58 (1998) 024008

// Instead of integrating mass(radius) and pressure(radius) Lindblom introduces
// the variables u and v, where u=radius^2 and v=mass/radius. The integration
// is then done with the log of the specific enthalpy (log(h)) as the
// independent variable.

// Confusingly, Lindblom's paper simply labels the independent variable as h.
// The h
// in Lindblom's paper is NOT the specific enthalpy. Rather, Lindblom's h is in
// fact log(h).

namespace tov {

template <bool IsRelativistic, size_t dim>
void lindblom(const state_type& u_and_v, state_type& dudh_and_dvdh,
              const double h,
              const std::unique_ptr<EquationsOfState::EquationOfState<
                  IsRelativistic, dim>>& poly) noexcept {
  const double& u = u_and_v[0];
  const double& v = u_and_v[1];
  double& dudh = dudh_and_dvdh[0];
  double& dvdh = dudh_and_dvdh[1];

  if (h > 0.0) {
    if ((u == 0.0) && (v == 0.0)) {
      Scalar<double> central_specific_enthalpy{std::exp(h)};

      Scalar<double> central_rest_mass_density{
          poly->rest_mass_density_from_enthalpy(central_specific_enthalpy)};

      Scalar<double> central_pressure{
          poly->pressure_from_density(central_rest_mass_density)};

      Scalar<double> central_energy_density{get(central_specific_enthalpy) *
                                                get(central_rest_mass_density) -
                                            get(central_pressure)};

      dudh = -3.0 / (2.0 * M_PI * (get(central_energy_density) +
                                   3.0 * get(central_pressure)));

      dvdh = -2.0 * get(central_energy_density) /
             (get(central_energy_density) + 3.0 * get(central_pressure));

    }

    else {
      Scalar<double> specific_enthalpy{std::exp(h)};

      Scalar<double> rest_mass_density{
          poly->rest_mass_density_from_enthalpy(specific_enthalpy)};

      Scalar<double> pressure{poly->pressure_from_density(rest_mass_density)};

      Scalar<double> energy_density{
          get(specific_enthalpy) * get(rest_mass_density) - get(pressure)};

      dudh = -2.0 * u * (1.0 - 2.0 * v) / (4.0 * M_PI * u * get(pressure) + v);

      dvdh = -(1.0 - 2.0 * v) * (4.0 * M_PI * u * get(energy_density) - v) /
             (4.0 * M_PI * u * get(pressure) + v);
    }
  }

  else {
    return;
  }
}

template <bool IsRelativistic, size_t dim>
InterpolationOutput TOV_Output::tov_solver(
    std::unique_ptr<EquationsOfState::EquationOfState<IsRelativistic, dim>>&
        polyM,
    double central_mass_density_in) noexcept {
  state_type u_and_v = {0.0, 0.0};

  Scalar<double> central_mass_density{central_mass_density_in};

  double hc = std::log(
      get(polyM->specific_enthalpy_from_density(central_mass_density)));

  typedef boost::numeric::odeint::runge_kutta_dopri5<state_type> dopri5_type;
  typedef boost::numeric::odeint::controlled_runge_kutta<dopri5_type>
      controlled_dopri5_type;
  typedef boost::numeric::odeint::dense_output_runge_kutta<
      controlled_dopri5_type>
      dense_output_dopri5_type;

  dense_output_dopri5_type dopri5 =
      make_dense_output(1.0e-14, 1.0e-20, dopri5_type());

  Observer observer{};

  boost::numeric::odeint::integrate_adaptive(
      dopri5,
      [&polyM](state_type& lindblom_u_and_v, state_type& lindblom_dudh_and_dvdh,
               double lindblom_enthalpy) {
        return lindblom(lindblom_u_and_v, lindblom_dudh_and_dvdh,
                        lindblom_enthalpy, polyM);
      },
      u_and_v, hc, 0.000, -1.0e-10, std::ref(observer));

  InterpolationOutput interout(observer.radius, observer.mass,
                               observer.log_enthalpy);

  return interout;
}

template <bool IsRelativistic, size_t dim>
InterpolationOutput TOV_Output::tov_solver_for_testing(
    std::unique_ptr<EquationsOfState::EquationOfState<IsRelativistic, dim>>&
        polyM,
    double central_mass_density_in, double h_final) noexcept {
  state_type u_and_v = {0.0, 0.0};

  Scalar<double> central_mass_density{central_mass_density_in};

  double hc = std::log(
      get(polyM->specific_enthalpy_from_density(central_mass_density)));

  typedef boost::numeric::odeint::runge_kutta_dopri5<state_type> dopri5_type;
  typedef boost::numeric::odeint::controlled_runge_kutta<dopri5_type>
      controlled_dopri5_type;
  typedef boost::numeric::odeint::dense_output_runge_kutta<
      controlled_dopri5_type>
      dense_output_dopri5_type;

  dense_output_dopri5_type dopri5 =
      make_dense_output(1.0e-14, 1.0e-20, dopri5_type());

  Observer observer{};

  boost::numeric::odeint::integrate_adaptive(
      dopri5,
      [&polyM](state_type& lindblom_u_and_v, state_type& lindblom_dudh_and_dvdh,
               double lindblom_enthalpy) {
        return lindblom(lindblom_u_and_v, lindblom_dudh_and_dvdh,
                        lindblom_enthalpy, polyM);
      },
      u_and_v, hc, h_final, -1.0e-10, std::ref(observer));

  InterpolationOutput interout(observer.radius, observer.mass,
                               observer.log_enthalpy);

  return interout;
}

template InterpolationOutput TOV_Output::tov_solver(
    std::unique_ptr<EquationsOfState::EquationOfState<true, 1>>& polyM,
    double central_mass_density_in);

template InterpolationOutput TOV_Output::tov_solver(
    std::unique_ptr<EquationsOfState::EquationOfState<false, 1>>& polyM,
    double central_mass_density_in);

template InterpolationOutput TOV_Output::tov_solver_for_testing(
    std::unique_ptr<EquationsOfState::EquationOfState<true, 1>>& polyM,
    double central_mass_density_in, double h_final);

template InterpolationOutput TOV_Output::tov_solver_for_testing(
    std::unique_ptr<EquationsOfState::EquationOfState<false, 1>>& polyM,
    double central_mass_density_in, double h_final);

}  // end of namespace tov
