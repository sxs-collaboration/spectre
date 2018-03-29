// Distributed under the MIT License.
// See LICENSE.txt for details.


#include "PointwiseFunctions/EquationsOfState/ConstDensity.hpp"
#include <boost/array.hpp>
#include <boost/math/tools/roots.hpp>
#include <boost/numeric/odeint.hpp>
#include <cmath>
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/EquationsOfState/EOS.hpp"

Scalar<double> EquationsOfState::ConstDensity::baryon_density(
    const Scalar<double>& log_specific_enthalpy) {
  return Scalar<double>{central_energy_density *
                        exp(get(log_specific_enthalpy))};
}

Scalar<double> EquationsOfState::ConstDensity::log_specific_enthalpy(
    const Scalar<double>& baryon_density) {
  return Scalar<double>{log(get(baryon_density) / central_energy_density)};
}

Scalar<double> EquationsOfState::ConstDensity::pressure(
    const Scalar<double>& log_specific_enthalpy) {
  return Scalar<double>{central_energy_density *
                        expm1(get(log_specific_enthalpy))};
}
