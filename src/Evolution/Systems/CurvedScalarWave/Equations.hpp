// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
template <typename X, typename Symm, typename IndexList>
class Tensor;

namespace CurvedScalarWave {
struct Psi;
struct Pi;
template <size_t Dim>
struct Phi;
struct ConstraintGamma1;
struct ConstraintGamma2;
}  // namespace CurvedScalarWave
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
 * \partial_t\psi = & (1 + \gamma_1) N^k \partial_k \psi - N \Pi - \gamma_1 N^k
 * \Phi_k \\
 *
 * \partial_t\Pi = & - N g^{ij}\partial_i\Phi_j + N^k \partial_k \Pi + \gamma_1
 * \gamma_2 N^k \partial_k \psi  + N \Gamma^i - g^{ij} \Phi_i \partial_j N
 *  + N K \Pi - \gamma_1  \gamma_2 N^k \Phi_k\\
 *
 * \partial_t\Phi_i = & - N \partial_i \Pi  + N^k \partial_k \Phi + \gamma_2
 * N \partial_i \psi - \Pi
 * \partial_i N + \Phi_k \partial_i N^j - \gamma_2 N \Phi_i\\
 * \f}
 *
 * where \f$\psi\f$ is the scalar field, \f$\Pi\f$ is the
 * conjugate momentum to \f$\psi\f$, \f$\Phi_i=\partial_i\psi\f$ is an
 * auxiliary variable, \f$N\f$ is the lapse, \f$N^k\f$ is the shift, \f$ g_{ij}
 * \f$ is the spatial metric, \f$ K \f$ is the trace of the extrinsic curvature,
 * and \f$ \Gamma^i \f$ is the trace of the Christoffel symbol of the second
 * kind.
 * \f$\gamma_1, \gamma_2\f$ are constraint damping parameters.
 */
template <size_t Dim>
struct ComputeDuDt {
  using argument_tags = tmpl::list<
      Pi, Phi<Dim>, Tags::deriv<Psi, tmpl::size_t<Dim>, Frame::Inertial>,
      Tags::deriv<Pi, tmpl::size_t<Dim>, Frame::Inertial>,
      Tags::deriv<Phi<Dim>, tmpl::size_t<Dim>, Frame::Inertial>,
      gr::Tags::Lapse<DataVector>,
      gr::Tags::Shift<Dim, Frame::Inertial, DataVector>,
      Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<Dim>,
                  Frame::Inertial>,
      Tags::deriv<gr::Tags::Shift<Dim, Frame::Inertial, DataVector>,
                  tmpl::size_t<Dim>, Frame::Inertial>,
      gr::Tags::InverseSpatialMetric<Dim, Frame::Inertial, DataVector>,
      gr::Tags::TraceSpatialChristoffelSecondKind<Dim, Frame::Inertial,
                                                  DataVector>,
      gr::Tags::TraceExtrinsicCurvature<DataVector>, ConstraintGamma1,
      ConstraintGamma2>;

  static void apply(
      gsl::not_null<Scalar<DataVector>*> dt_pi,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*> dt_phi,
      gsl::not_null<Scalar<DataVector>*> dt_psi, const Scalar<DataVector>& pi,
      const tnsr::i<DataVector, Dim>& phi,
      const tnsr::i<DataVector, Dim>& d_psi,
      const tnsr::i<DataVector, Dim>& d_pi,
      const tnsr::ij<DataVector, Dim>& d_phi, const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, Dim>& shift,
      const tnsr::i<DataVector, Dim>& deriv_lapse,
      const tnsr::iJ<DataVector, Dim>& deriv_shift,
      const tnsr::II<DataVector, Dim>& upper_spatial_metric,
      const tnsr::I<DataVector, Dim>& trace_spatial_christoffel,
      const Scalar<DataVector>& trace_extrinsic_curvature,
      const Scalar<DataVector>& gamma1,
      const Scalar<DataVector>& gamma2) noexcept;
};
}  // namespace CurvedScalarWave
