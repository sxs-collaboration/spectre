// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class template GeneralizedHarmonicEquations.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/TagsDeclarations.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"  // IWYU pragma: keep
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;

namespace Tags {
template <typename>
class dt;
template <typename>
struct NormalDotFlux;
}  // namespace Tags

namespace gsl {
template <class T>
class not_null;
}  // namespace gsl
/// \endcond

namespace GeneralizedHarmonic {
/*!
 * \brief Compute the RHS of the Generalized Harmonic formulation of
 * Einstein's equations.
 *
 * \details For the full form of the equations see \cite Lindblom2005qh.
 */
template <size_t Dim>
struct ComputeDuDt {
 public:
  using argument_tags = tmpl::list<
      gr::Tags::SpacetimeMetric<Dim>, Tags::Pi<Dim>, Tags::Phi<Dim>,
      ::Tags::deriv<gr::Tags::SpacetimeMetric<Dim>, tmpl::size_t<Dim>,
                    Frame::Inertial>,
      ::Tags::deriv<Tags::Pi<Dim>, tmpl::size_t<Dim>, Frame::Inertial>,
      ::Tags::deriv<Tags::Phi<Dim>, tmpl::size_t<Dim>, Frame::Inertial>,
      Tags::ConstraintGamma0, Tags::ConstraintGamma1, Tags::ConstraintGamma2,
      Tags::GaugeH<Dim>, Tags::SpacetimeDerivGaugeH<Dim>, gr::Tags::Lapse<>,
      gr::Tags::Shift<Dim>, gr::Tags::InverseSpatialMetric<Dim>,
      gr::Tags::InverseSpacetimeMetric<Dim>,
      gr::Tags::TraceSpacetimeChristoffelFirstKind<Dim>,
      gr::Tags::SpacetimeChristoffelFirstKind<Dim>,
      gr::Tags::SpacetimeChristoffelSecondKind<Dim>,
      gr::Tags::SpacetimeNormalVector<Dim>,
      gr::Tags::SpacetimeNormalOneForm<Dim>>;

  static void apply(
      gsl::not_null<tnsr::aa<DataVector, Dim>*> dt_spacetime_metric,
      gsl::not_null<tnsr::aa<DataVector, Dim>*> dt_pi,
      gsl::not_null<tnsr::iaa<DataVector, Dim>*> dt_phi,
      const tnsr::aa<DataVector, Dim>& spacetime_metric,
      const tnsr::aa<DataVector, Dim>& pi,
      const tnsr::iaa<DataVector, Dim>& phi,
      const tnsr::iaa<DataVector, Dim>& d_spacetime_metric,
      const tnsr::iaa<DataVector, Dim>& d_pi,
      const tnsr::ijaa<DataVector, Dim>& d_phi,
      const Scalar<DataVector>& gamma0, const Scalar<DataVector>& gamma1,
      const Scalar<DataVector>& gamma2,
      const tnsr::a<DataVector, Dim>& gauge_function,
      const tnsr::ab<DataVector, Dim>& spacetime_deriv_gauge_function,
      const Scalar<DataVector>& lapse, const tnsr::I<DataVector, Dim>& shift,
      const tnsr::II<DataVector, Dim>& inverse_spatial_metric,
      const tnsr::AA<DataVector, Dim>& inverse_spacetime_metric,
      const tnsr::a<DataVector, Dim>& trace_christoffel,
      const tnsr::abb<DataVector, Dim>& christoffel_first_kind,
      const tnsr::Abb<DataVector, Dim>& christoffel_second_kind,
      const tnsr::A<DataVector, Dim>& normal_spacetime_vector,
      const tnsr::a<DataVector, Dim>& normal_spacetime_one_form);
};

/*!
 * \brief Compute the fluxes of the Generalized Harmonic formulation of
 * Einstein's equations.
 *
 * \details The expressions for the fluxes is obtained from
 * \cite Lindblom2005qh.
 * The fluxes for each variable are obtained by taking the principal part of
 * equations 35, 36, and 37, and replacing derivatives \f$ \partial_k \f$
 * with the unit normal \f$ n_k \f$. This gives:
 *
 * \f{align*}
 * F(\psi_{ab}) =& -(1 + \gamma_1) N^k n_k \psi_{ab} \\
 * F(\Pi_{ab}) =& - N^k n_k \Pi_{ab} + N g^{ki}n_k \Phi_{iab} - \gamma_1
 * \gamma_2
 * N^k n_k \psi_{ab} \\
 * F(\Phi_{iab}) =& - N^k n_k \Phi_{iab} + N n_i \Pi_{ab} - \gamma_1 \gamma_2
 * N^i \Phi_{iab}
 * \f}
 *
 * where \f$\psi_{ab}\f$ is the spacetime metric, \f$\Pi_{ab}\f$ its conjugate
 * momentum, \f$ \Phi_{iab} \f$ is an auxiliary field as defined by the tag Phi,
 * \f$N\f$ is the lapse, \f$ N^k \f$ is the shift, \f$ g^{ki} \f$ is the inverse
 * spatial metric, and \f$ \gamma_1, \gamma_2 \f$ are constraint damping
 * parameters.
 */
template <size_t Dim>
struct ComputeNormalDotFluxes {
 public:
  using argument_tags =
      tmpl::list<gr::Tags::SpacetimeMetric<Dim>, Tags::Pi<Dim>, Tags::Phi<Dim>,
                 Tags::ConstraintGamma1, Tags::ConstraintGamma2,
                 gr::Tags::Lapse<>, gr::Tags::Shift<Dim>,
                 gr::Tags::InverseSpatialMetric<Dim>>;

  static void apply(
      gsl::not_null<tnsr::aa<DataVector, Dim>*>
          spacetime_metric_normal_dot_flux,
      gsl::not_null<tnsr::aa<DataVector, Dim>*> pi_normal_dot_flux,
      gsl::not_null<tnsr::iaa<DataVector, Dim>*> phi_normal_dot_flux,
      const tnsr::aa<DataVector, Dim>& spacetime_metric,
      const tnsr::aa<DataVector, Dim>& pi,
      const tnsr::iaa<DataVector, Dim>& phi, const Scalar<DataVector>& gamma1,
      const Scalar<DataVector>& gamma2, const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, Dim>& shift,
      const tnsr::II<DataVector, Dim>& inverse_spatial_metric,
      const tnsr::i<DataVector, Dim>& unit_normal) noexcept;
};
}  // namespace GeneralizedHarmonic
