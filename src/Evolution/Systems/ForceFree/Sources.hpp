// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
/// \endcond

namespace ForceFree {

namespace detail {
void sources_impl(
    gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> source_tilde_e,
    gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> source_tilde_b,
    gsl::not_null<Scalar<DataVector>*> source_tilde_psi,
    gsl::not_null<Scalar<DataVector>*> source_tilde_phi,

    // temp variables
    const tnsr::I<DataVector, 3, Frame::Inertial>&
        trace_spatial_christoffel_second,

    // EM args
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_e,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const Scalar<DataVector>& tilde_psi, const Scalar<DataVector>& tilde_phi,
    const Scalar<DataVector>& tilde_q,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_j_drift,
    double kappa_psi, double kappa_phi,
    // GR args
    const Scalar<DataVector>& lapse,
    const tnsr::i<DataVector, 3, Frame::Inertial>& d_lapse,
    const tnsr::iJ<DataVector, 3, Frame::Inertial>& d_shift,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& extrinsic_curvature);
}  // namespace detail

/*!
 * \brief Compute the source terms for the GRFFE system with divergence
 * cleaning.
 *
 * \f{align*}
 *  S(\tilde{E}^i) &= -\alpha \sqrt{\gamma} J^i - \tilde{E}^j \partial_j \beta^i
 *    + \tilde{\psi} ( \gamma^{ij} \partial_j \alpha - \alpha \gamma^{jk}
 *    \Gamma^i_{jk} ) \\
 *  S(\tilde{B}^i) &= -\tilde{B}^j \partial_j \beta^i + \tilde{\phi} (
 *    \gamma^{ij} \partial_j \alpha - \alpha \gamma^{jk} \Gamma^i_{jk} ) \\
 *  S(\tilde{\psi}) &= \tilde{E}^k \partial_k \alpha + \alpha \tilde{q} - \alpha
 *    \tilde{\phi} ( K + \kappa_\phi ) \\
 *  S(\tilde{\phi}) &= \tilde{B}^k \partial_k \alpha - \alpha \tilde{\phi} (K +
 *    \kappa_\phi ) \\
 *  S(\tilde{q}) &= 0
 * \f}
 *
 * where the conserved variables \f$\tilde{E}^i, \tilde{B}^i, \tilde{\psi},
 * \tilde{\phi}, \tilde{q}\f$ are densitized electric field, magnetic field,
 * magnetic divergence cleaning field, electric divergence cleaning field, and
 * electric charge density.
 *
 * \f$J^i\f$ is the spatial electric current density, \f$\alpha\f$ is the lapse,
 * \f$\beta^i\f$ is the shift, \f$\gamma^{ij}\f$ is the spatial metric,
 * \f$\gamma\f$ is the determinant of spatial metric, \f$\Gamma^i_{jk}\f$ is the
 * spatial Christoffel symbol, \f$K\f$ is the trace of extrinsic curvature.
 * \f$\kappa_\phi\f$ and \f$\kappa_\psi\f$ are damping parameters associated
 * with divergence cleaning of magnetic and electric fields, respectively.
 *
 */
struct Sources {
  using return_tags =
      tmpl::list<::Tags::Source<Tags::TildeE>, ::Tags::Source<Tags::TildeB>,
                 ::Tags::Source<Tags::TildePsi>,
                 ::Tags::Source<Tags::TildePhi>>;

  using argument_tags = tmpl::list<
      // EM variables
      Tags::TildeE, Tags::TildeB, Tags::TildePsi, Tags::TildePhi, Tags::TildeQ,
      Tags::KappaPsi, Tags::KappaPhi, Tags::ParallelConductivity,
      // GR variables
      gr::Tags::Lapse<DataVector>,
      ::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                    Frame::Inertial>,
      ::Tags::deriv<gr::Tags::Shift<DataVector, 3>, tmpl::size_t<3>,
                    Frame::Inertial>,
      ::Tags::deriv<gr::Tags::SpatialMetric<DataVector, 3>, tmpl::size_t<3>,
                    Frame::Inertial>,
      gr::Tags::SpatialMetric<DataVector, 3>,
      gr::Tags::InverseSpatialMetric<DataVector, 3>,
      gr::Tags::SqrtDetSpatialMetric<DataVector>,
      gr::Tags::ExtrinsicCurvature<DataVector, 3>>;

  static void apply(
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> source_tilde_e,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> source_tilde_b,
      gsl::not_null<Scalar<DataVector>*> source_tilde_psi,
      gsl::not_null<Scalar<DataVector>*> source_tilde_phi,
      // EM variables
      const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_e,
      const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
      const Scalar<DataVector>& tilde_psi, const Scalar<DataVector>& tilde_phi,
      const Scalar<DataVector>& tilde_q, double kappa_psi, double kappa_phi,
      double parallel_conductivity,
      // GR variables
      const Scalar<DataVector>& lapse,
      const tnsr::i<DataVector, 3, Frame::Inertial>& d_lapse,
      const tnsr::iJ<DataVector, 3, Frame::Inertial>& d_shift,
      const tnsr::ijj<DataVector, 3, Frame::Inertial>& d_spatial_metric,
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
      const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
      const Scalar<DataVector>& sqrt_det_spatial_metric,
      const tnsr::ii<DataVector, 3, Frame::Inertial>& extrinsic_curvature);
};

}  // namespace ForceFree
