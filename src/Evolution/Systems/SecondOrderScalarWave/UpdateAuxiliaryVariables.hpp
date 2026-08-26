// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/SecondOrderScalarWave/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace SecondOrderScalarWave {
/*!
 * \brief Computes the auxiliary variable \f$\Phi_i = \partial_i \Psi\f$
 *
 * \details In the LDG scheme, the auxiliary variable \f$\Phi_i\f$ is not
 * evolved in time but is recomputed from \f$\Psi\f$ at each substep.
 * After this mutator runs, the boundary correction from
 * `ApplyAuxiliaryBoundaryCorrectionsToVariables` will correct \f$\Phi_i\f$
 * at element interfaces.
 */
template <size_t Dim>
struct UpdateAuxiliaryVariables {
  using return_tags = tmpl::list<Tags::Phi<Dim>>;
  using argument_tags =
      tmpl::list<Tags::Psi, domain::Tags::Mesh<Dim>,
                 domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
                                               Frame::Inertial>>;

  static void apply(
      const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*> phi,
      const Scalar<DataVector>& psi, const Mesh<Dim>& mesh,
      const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                            Frame::Inertial>& inverse_jacobian) {
    partial_derivative(phi, psi, mesh, inverse_jacobian);
  }
};
}  // namespace SecondOrderScalarWave
