// Distributed under the MIT License.
// See LICENSE.txt for details.


#include <boost/array.hpp>
#include <boost/math/tools/roots.hpp>
#include <boost/numeric/odeint.hpp>
#include <cmath>
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#pragma once

namespace EquationsOfState {}

class EquationOfState

{
 public:
  virtual Scalar<double> baryon_density(
      const Scalar<double>& log_specific_enthalpy) = 0;

  virtual Scalar<double> pressure(
      const Scalar<double>& log_specific_enthalpy) = 0;

  virtual Scalar<double> energy_density(
      const Scalar<double>& log_specific_enthalpy) = 0;

  virtual Scalar<double> log_specific_enthalpy(
      const Scalar<double>& baryon_density) = 0;

  virtual ~EquationOfState() = default;
};
