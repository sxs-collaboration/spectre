// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/FaceNormal.hpp"
#include "Evolution/Systems/CurvedScalarWave/Characteristics.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Protocols.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
template <typename X, typename Symm, typename IndexList>
class Tensor;
/// \endcond

namespace CurvedScalarWave {
/*!
 * \brief Compute fluxes for the scalar-wave system in curved spacetime.
 *
 * \details The expressions for fluxes is obtained from \cite Holst2004wt by
 * taking the principal part of equations 15, 23, and 24, and replacing
 * derivatives \f$ \partial_k \f$ with the unit normal \f$ n_k \f$ (c.f. Gauss'
 * theorem). This gives:
 *
 * \f{align*}
 * F(\psi) &= -(1 + \gamma_1) \beta^k n_k \psi \\
 * F(\Pi) &= - \beta^k n_k \Pi + \alpha g^{ki}n_k \Phi_{i}
 *           - \gamma_1 \gamma_2 \beta^k n_k \Psi  \\
 * F(\Phi_{i}) &= - \beta^k n_k \Phi_{i} + \alpha n_i \Pi - \gamma_2 \alpha n_i
 * \psi \f}
 *
 * where \f$\psi\f$ is the scalar field, \f$\Pi\f$ is its conjugate
 * momentum, \f$ \Phi_{i} \f$ is an auxiliary field defined as the spatial
 * derivative of \f$\psi\f$, \f$\alpha\f$ is the lapse, \f$ \beta^k \f$ is the
 * shift, \f$ g^{ki} \f$ is the inverse spatial metric, and \f$ \gamma_1,
 * \gamma_2\f$ are constraint damping parameters. Note that the last term in
 * \f$F(\Pi)\f$ will be identically zero, as it is necessary to set
 * \f$\gamma_1\gamma_2=0\f$ in order to make this first-order formulation of the
 * curved scalar wave system symmetric hyperbolic.
 */
template <size_t Dim>
struct ComputeNormalDotFluxes {
 public:
  using argument_tags =
      tmpl::list<Pi, Phi<Dim>, Psi, Tags::ConstraintGamma1,
                 Tags::ConstraintGamma2, gr::Tags::Lapse<>,
                 gr::Tags::Shift<Dim>, gr::Tags::InverseSpatialMetric<Dim>,
                 ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<
                     Dim, Frame::Inertial>>>;

  static void apply(
      gsl::not_null<Scalar<DataVector>*> pi_normal_dot_flux,
      gsl::not_null<tnsr::i<DataVector, Dim>*> phi_normal_dot_flux,
      gsl::not_null<Scalar<DataVector>*> psi_normal_dot_flux,
      const Scalar<DataVector>& pi, const tnsr::i<DataVector, Dim>& phi,
      const Scalar<DataVector>& psi, const Scalar<DataVector>& gamma1,
      const Scalar<DataVector>& gamma2, const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, Dim>& shift,
      const tnsr::II<DataVector, Dim>& inverse_spatial_metric,
      const tnsr::i<DataVector, Dim>& interface_unit_normal) noexcept;
};
}  // namespace CurvedScalarWave
