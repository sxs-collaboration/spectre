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

class DataVector;

namespace EquationsOfState {

class Polytrope : public EquationOfState {
 public:
  Scalar<double> baryon_density(
      const Scalar<double>& log_specific_enthalpy) override;

  Scalar<double> pressure(const Scalar<double>& log_specific_enthalpy) override;

  Scalar<double> energy_density(
      const Scalar<double>& log_specific_enthalpy) override;

  Scalar<double> log_specific_enthalpy(
      const Scalar<double>& baryon_density) override;

  Polytrope(const double K, const double gam);

 private:
  const double gas_constant = std::numeric_limits<double>::signaling_NaN();

  const double gamma = std::numeric_limits<double>::signaling_NaN();
};

}  // namespace EquationsOfState
