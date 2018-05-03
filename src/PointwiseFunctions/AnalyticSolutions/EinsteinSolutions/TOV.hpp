// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <boost/array.hpp>
#include <boost/math/interpolators/barycentric_rational.hpp>
#include <boost/math/tools/roots.hpp>
#include <boost/numeric/odeint.hpp>
#include <boost/range/adaptors.hpp>

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"        // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"     // IWYU pragma: keep
#include "PointwiseFunctions/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/EquationsOfState/PolytropicFluid.hpp"

#pragma once

namespace tov {

using state_type = std::array<double, 2>;

template <bool IsRelativistic, size_t dim>
void lindblom(
    const state_type& u_and_v, state_type& dudh_and_dvdh, double h,
    const std::unique_ptr<
        EquationsOfState::EquationOfState<IsRelativistic, dim>>& poly) noexcept;

class Observer {
 public:
  void operator()(const state_type& x, double h) {
    radius.push_back(std::sqrt(x[0]));
    mass.push_back(std::sqrt(x[0]) * x[1]);
    log_enthalpy.push_back(h);
  }

  std::vector<double> radius;
  std::vector<double> mass;
  std::vector<double> log_enthalpy;
};

class InterpolationOutput {
 public:
  InterpolationOutput(const std::vector<double>& radius,
                      const std::vector<double>& mass,
                      const std::vector<double>& log_enthalpy)
      : radius_(radius.back()),
        interpolated_mass_(radius.data(), mass.data(), mass.size(), 5),
        interpolated_log_enthalpy_(radius.data(), log_enthalpy.data(),
                                   log_enthalpy.size(), 3) {}

  double final_radius() noexcept { return radius_; }
  double mass_from_radius(double r) noexcept { return interpolated_mass_(r); }
  double specific_enthalpy_from_radius(double r) noexcept {
    return std::exp(interpolated_log_enthalpy_(r));
  }
  double log_specific_enthalpy_from_radius(double r) noexcept {
    return interpolated_log_enthalpy_(r);
  }

 private:
  double radius_;
  boost::math::barycentric_rational<double> interpolated_mass_;
  boost::math::barycentric_rational<double> interpolated_log_enthalpy_;
};

class TOV_Output {
 public:
  template <bool IsRelativistic, size_t dim>
  InterpolationOutput tov_solver(
      std::unique_ptr<EquationsOfState::EquationOfState<IsRelativistic, dim>>&
          polyM,
      double central_mass_density_in) noexcept;

  template <bool IsRelativistic, size_t dim>
  InterpolationOutput tov_solver_for_testing(
      std::unique_ptr<EquationsOfState::EquationOfState<IsRelativistic, dim>>&
          polyM,
      double central_mass_density_in, double h_final) noexcept;
};

}  // end of namespace tov
