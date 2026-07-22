// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/DiscontinuousGalerkin/TimeDerivativeDecisions.hpp"
#include "Evolution/Systems/SecondOrderScalarWave/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl

class DataVector;
/// \endcond

namespace SecondOrderScalarWave {
/*!
 * \brief Compute the time derivatives for the second-order scalar wave system
 */
template <size_t Dim>
struct TimeDerivative {
  using temporary_tags = tmpl::list<>;

  using argument_tags = tmpl::list<Tags::Pi>;

  static evolution::dg::TimeDerivativeDecisions<Dim> apply(
      gsl::not_null<Scalar<DataVector>*> dt_psi,
      gsl::not_null<Scalar<DataVector>*> dt_pi,

      // Partial derivative arguments. Listed in the system struct as
      // gradient_variables. Only the derivative of Phi enters the time
      // derivatives; the evolved variables' derivatives are in
      // gradient_variables solely for the framework's moving-mesh term.
      const tnsr::i<DataVector, Dim, Frame::Inertial>& /*d_psi*/,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& /*d_pi*/,
      const tnsr::ij<DataVector, Dim, Frame::Inertial>& d_phi,

      // Terms list in argument_tags above
      const Scalar<DataVector>& pi);
};
}  // namespace SecondOrderScalarWave
