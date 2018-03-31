// Distributed under the MIT License.
// See LICENSE.txt for details.


#include <boost/array.hpp>
#include <boost/math/tools/roots.hpp>
#include <boost/numeric/odeint.hpp>
#include <cmath>
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/EquationsOfState/EOS.hpp"
#pragma once

namespace EquationsOfState {

class ConstDensity : public EquationOfState {
  Scalar<double> baryon_density(
      const Scalar<double>& log_specific_enthalpy) override;

  Scalar<double> pressure(const Scalar<double>& log_specific_enthalpy) override;

  Scalar<double> log_specific_enthalpy(
      const Scalar<double>& baryon_density) override;

  const double central_baryon_density = 1.5e-03;

  const double central_energy_density = 1.0e-03;

  const double energy_density = central_energy_density;
};

}  // namespace EquationsOfState
