// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/ScalarWave/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl

class DataVector;
/// \endcond

namespace ScalarWave {
/*!
 * \brief Compute the time derivatives for scalar wave system
 */
template <size_t Dim>
struct TimeDerivative {
  using temporary_tags = tmpl::list<Tags::ConstraintGamma2>;
  using argument_tags = tmpl::list<Pi, Phi<Dim>, Tags::ConstraintGamma2>;

  static void apply(
      // Time derivatives returned by reference. All the tags in the
      // variables_tag in the system struct.
      gsl::not_null<Scalar<DataVector>*> dt_pi,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*> dt_phi,
      gsl::not_null<Scalar<DataVector>*> dt_psi,

      gsl::not_null<Scalar<DataVector>*> result_gamma2,

      // Partial derivative arguments. Listed in the system struct as
      // gradient_variables.
      const tnsr::i<DataVector, Dim, Frame::Inertial>& d_pi,
      const tnsr::ij<DataVector, Dim, Frame::Inertial>& d_phi,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& d_psi,

      // Terms list in argument_tags above
      const Scalar<DataVector>& pi,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& phi,
      const Scalar<DataVector>& gamma2) noexcept;
};
}  // namespace ScalarWave
