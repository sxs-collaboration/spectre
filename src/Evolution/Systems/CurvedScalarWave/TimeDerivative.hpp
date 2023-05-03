// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/CurvedScalarWave/Characteristics.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
template <typename X, typename Symm, typename IndexList>
class Tensor;
/// \endcond

namespace CurvedScalarWave {
/*!
 * \brief Compute the time derivative of the evolved variables of the
 * first-order scalar wave system on a curved background.
 *
 * The evolution equations for the first-order scalar wave system are given by
 * \cite Holst2004wt :
 *
 * \f{align}
 * \partial_t\Psi = & - \alpha \Pi + \beta^k \partial_k \Psi
 * + \gamma_1 \beta^k (\partial_k \Psi - \Phi_k) \\
 *
 * \partial_t\Pi = & \alpha K \Pi + \beta^i \partial_i \Pi
 * + \alpha \Gamma^i \Phi_i
 * + \gamma_1 \gamma_2 \beta^i ( \partial_i \Psi - \Phi_i )
 * - \alpha \gamma^{ij} \partial_i\Phi_j
 * - \gamma^{ij} \Phi_i \partial_j \alpha \\
 *
 * \partial_t\Phi_i = & - \alpha \partial_i \Pi  + \beta^k \partial_k \Phi_i
 * + \gamma_2 \alpha ( \partial_i \Psi - \Phi_i )
 * - \Pi \partial_i \alpha + \Phi_j \partial_i \beta^j \\
 * \f}
 *
 * where \f$\Psi\f$ is the scalar field, \f$\Pi\f$ is the
 * conjugate momentum to \f$\Psi\f$, \f$\Phi_i=\partial_i\Psi\f$ is an
 * auxiliary variable, \f$\alpha\f$ is the lapse, \f$\beta^k\f$ is the shift,
 * \f$ \gamma_{ij} \f$ is the spatial metric, \f$ K \f$ is the trace of the
 * extrinsic curvature, and \f$ \Gamma^i \f$ is the trace of the spatial
 * Christoffel symbol of the second kind. \f$\gamma_1, \gamma_2\f$ are
 * constraint damping parameters.
 */
template <size_t Dim>
struct TimeDerivative {
 public:
  using temporary_tags =
      tmpl::list<gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, Dim>,
                 gr::Tags::InverseSpatialMetric<DataVector, Dim>,
                 Tags::ConstraintGamma1, Tags::ConstraintGamma2>;

  using argument_tags =
      tmpl::list<Tags::Pi, Tags::Phi<Dim>, gr::Tags::Lapse<DataVector>,
                 gr::Tags::Shift<DataVector, Dim>,
                 ::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<Dim>,
                               Frame::Inertial>,
                 ::Tags::deriv<gr::Tags::Shift<DataVector, Dim>,
                               tmpl::size_t<Dim>, Frame::Inertial>,
                 gr::Tags::InverseSpatialMetric<DataVector, Dim>,
                 gr::Tags::TraceSpatialChristoffelSecondKind<DataVector, Dim>,
                 gr::Tags::TraceExtrinsicCurvature<DataVector>,
                 Tags::ConstraintGamma1, Tags::ConstraintGamma2>;

  static void apply(
      gsl::not_null<Scalar<DataVector>*> dt_psi,
      gsl::not_null<Scalar<DataVector>*> dt_pi,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*> dt_phi,

      gsl::not_null<Scalar<DataVector>*> result_lapse,
      gsl::not_null<tnsr::I<DataVector, Dim>*> result_shift,
      gsl::not_null<tnsr::II<DataVector, Dim>*> result_inverse_spatial_metric,
      gsl::not_null<Scalar<DataVector>*> result_gamma1,
      gsl::not_null<Scalar<DataVector>*> result_gamma2,

      const tnsr::i<DataVector, Dim>& d_psi,
      const tnsr::i<DataVector, Dim>& d_pi,
      const tnsr::ij<DataVector, Dim>& d_phi, const Scalar<DataVector>& pi,
      const tnsr::i<DataVector, Dim>& phi, const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, Dim>& shift,
      const tnsr::i<DataVector, Dim>& deriv_lapse,
      const tnsr::iJ<DataVector, Dim>& deriv_shift,
      const tnsr::II<DataVector, Dim>& upper_spatial_metric,
      const tnsr::I<DataVector, Dim>& trace_spatial_christoffel,
      const Scalar<DataVector>& trace_extrinsic_curvature,
      const Scalar<DataVector>& gamma1, const Scalar<DataVector>& gamma2);
};
}  // namespace CurvedScalarWave
