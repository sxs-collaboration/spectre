// Distributed under the MIT License.
// See LICENSE.txt for details.


#include "PointwiseFunctions/EquationsOfState/Polytrope.hpp"
#include <boost/array.hpp>
#include <boost/math/tools/roots.hpp>
#include <boost/numeric/odeint.hpp>
#include <cmath>
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/EquationsOfState/EOS.hpp"

EquationsOfState::Polytrope::Polytrope(const double K, const double gam)
    : gas_constant(K), gamma(gam) {}

Scalar<double> EquationsOfState::Polytrope::baryon_density(
    const Scalar<double>& log_specific_enthalpy) {
  return Scalar<double>{pow((gamma - 1.0) *
                                (expm1(get(log_specific_enthalpy))) /
                                (gas_constant * gamma),
                            1.0 / (gamma - 1.0))};
}

Scalar<double> EquationsOfState::Polytrope::log_specific_enthalpy(
    const Scalar<double>& baryon_density) {
  return Scalar<double>{log1p((gas_constant * gamma / (gamma - 1.0)) *
                              pow(get(baryon_density), (gamma - 1.0)))};
}

Scalar<double> EquationsOfState::Polytrope::pressure(
    const Scalar<double>& log_specific_enthalpy) {
  return Scalar<double>{gas_constant *
                        pow(get(baryon_density(log_specific_enthalpy)), gamma)};
}

Scalar<double> EquationsOfState::Polytrope::energy_density(
    const Scalar<double>& log_specific_enthalpy) {
  return Scalar<double>{
      get(baryon_density(log_specific_enthalpy)) +
      (1.0 / (gamma - 1.0)) * gas_constant *
          pow(get(baryon_density(log_specific_enthalpy)), gamma)};
}
