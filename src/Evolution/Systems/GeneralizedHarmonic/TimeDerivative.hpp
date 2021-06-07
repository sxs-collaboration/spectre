// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/DuDtTempTags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"  // IWYU pragma: keep
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;

namespace PUP {
class er;
}  // namespace PUP

namespace gsl {
template <class T>
class not_null;
}  // namespace gsl

template <typename, typename, typename>
class Tensor;
/// \endcond

// IWYU pragma: no_forward_declare Tags::deriv

namespace GeneralizedHarmonic {
/*!
 * \brief Compute the RHS of the Generalized Harmonic formulation of
 * Einstein's equations.
 *
 * The evolved variables are the spacetime metric \f$g_{ab}\f$, its spatial
 * derivative \f$\Phi_{iab}=\partial_i g_{ab}\f$, and conjugate momentum
 * \f$\Pi_{ab}=n^c\partial_c g_{ab}\f$, where \f$n^a\f$ is the spacetime
 * unit normal vector. The evolution equations are (Eqs. 35-57 of
 * \cite Lindblom2005qh)
 *
 * \f{align}{
 *   \partial_t g_{ab}-
 *   &\left(1+\gamma_1\right)\beta^k\partial_k g_{ab} =
 *     -\alpha \Pi_{ab}-\gamma_1\beta^i\Phi_{iab}, \\
 *
 *   \partial_t\Pi_{ab}-
 *   &\beta^k\partial_k\Pi_{ab} + \alpha \gamma^{ki}\partial_k\Phi_{iab}
 *     - \gamma_1\gamma_2\beta^k\partial_kg_{ab} \notag \\
 *   =&2\alpha g^{cd}\left(\gamma^{ij}\Phi_{ica}\Phi_{jdb}
 *      - \Pi_{ca}\Pi_{db} - g^{ef}\Gamma_{ace}\Gamma_{bdf}\right) \notag \\
 *   &-2\alpha \nabla_{(a}H_{b)}
 *     - \frac{1}{2}\alpha n^c n^d\Pi_{cd}\Pi_{ab}
 *     - \alpha n^c \Pi_{ci}\gamma^{ij}\Phi_{jab} \notag \\
 *   &+\alpha \gamma_0\left(2\delta^c{}_{(a} n_{b)}
 *     - (1 + \gamma_3)g_{ab}n^c\right)\mathcal{C}_c \notag \\
 *   &+ 2 \gamma_4 \alpha \Pi_{ab} n^c \mathcal{C}_c \notag \\
 *   &- \gamma_5\alpha n^c\mathcal{C}_c \left(\frac{\mathcal{C}_a\mathcal{C}_b
 *     - \frac{1}{2} g_{ab} \mathcal{C}_d \mathcal{C}^d}
 *     {\epsilon_{5} + 2 n^d \mathcal{C}_d n^e \mathcal{C}_e
 *     + \mathcal{C}_d \mathcal{C}^d} \right) \notag \\
 *   &-\gamma_1\gamma_2 \beta^i\Phi_{iab} \notag \\
 *   &-16\pi \alpha \left(T_{ab} - \frac{1}{2}g_{ab}T^c{}_c\right),\\
 *
 *   \partial_t\Phi_{iab}-
 *   &\beta^k\partial_k\Phi_{iab} + \alpha \partial_i\Pi_{ab}
 *     - \alpha \gamma_2\partial_ig_{ab} \notag \\
 *   =&\frac{1}{2}\alpha n^c n^d\Phi_{icd}\Pi_{ab}
 *      + \alpha \gamma^{jk}n^c\Phi_{ijc}\Phi_{kab} \notag \\
 *   &-\alpha \gamma_2\Phi_{iab},
 * \f}
 *
 * where \f$H_a\f$ is the gauge source function and
 * \f$\mathcal{C}_a=H_a+\Gamma_a\f$ is the gauge constraint. The constraint
 * damping parameters \f$\gamma_0\f$ \f$\gamma_1\f$, \f$\gamma_2\f$,
 * \f$\gamma_3\f$, \f$\gamma_4\f$, and \f$\gamma_5\f$ have units of inverse time
 * and control the time scales on which the constraints are damped to zero.
 *
 * \note We have not coded up the constraint damping terms for \f$\gamma_3\f$,
 * \f$\gamma_4\f$, and \f$\gamma_5\f$. \f$\gamma_3\f$ was found to be essential
 * for evolutions of black strings by Pretorius and Lehner \cite Lehner2010pn.
 */
template <size_t Dim>
struct TimeDerivative {
 public:
  using temporary_tags = tmpl::list<
      ::GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma1,
      ::GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma2,
      Tags::GaugeH<Dim>, Tags::SpacetimeDerivGaugeH<Dim>, Tags::Gamma1Gamma2,
      Tags::PiTwoNormals, Tags::NormalDotOneIndexConstraint, Tags::Gamma1Plus1,
      Tags::PiOneNormal<Dim>, Tags::GaugeConstraint<Dim, Frame::Inertial>,
      Tags::PhiTwoNormals<Dim>, Tags::ShiftDotThreeIndexConstraint<Dim>,
      Tags::PhiOneNormal<Dim>, Tags::PiSecondIndexUp<Dim>,
      Tags::ThreeIndexConstraint<Dim, Frame::Inertial>,
      Tags::PhiFirstIndexUp<Dim>, Tags::PhiThirdIndexUp<Dim>,
      Tags::SpacetimeChristoffelFirstKindThirdIndexUp<Dim>,
      gr::Tags::Lapse<DataVector>,
      gr::Tags::Shift<Dim, Frame::Inertial, DataVector>,
      gr::Tags::SpatialMetric<Dim, Frame::Inertial, DataVector>,
      gr::Tags::InverseSpatialMetric<Dim, Frame::Inertial, DataVector>,
      gr::Tags::DetSpatialMetric<DataVector>,
      gr::Tags::InverseSpacetimeMetric<Dim, Frame::Inertial, DataVector>,
      gr::Tags::SpacetimeChristoffelFirstKind<Dim, Frame::Inertial, DataVector>,
      gr::Tags::SpacetimeChristoffelSecondKind<Dim, Frame::Inertial,
                                               DataVector>,
      gr::Tags::TraceSpacetimeChristoffelFirstKind<Dim, Frame::Inertial,
                                                   DataVector>,
      gr::Tags::SpacetimeNormalVector<Dim, Frame::Inertial, DataVector>,
      gr::Tags::SpacetimeNormalOneForm<Dim, Frame::Inertial, DataVector>,
      gr::Tags::DerivativesOfSpacetimeMetric<Dim, Frame::Inertial, DataVector>>;
  using argument_tags = tmpl::list<
      gr::Tags::SpacetimeMetric<Dim>, Tags::Pi<Dim>, Tags::Phi<Dim>,
      ::GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma0,
      ::GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma1,
      ::GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma2,
      Tags::GaugeH<Dim>, Tags::SpacetimeDerivGaugeH<Dim>>;

  static void apply(
      gsl::not_null<tnsr::aa<DataVector, Dim>*> dt_spacetime_metric,
      gsl::not_null<tnsr::aa<DataVector, Dim>*> dt_pi,
      gsl::not_null<tnsr::iaa<DataVector, Dim>*> dt_phi,
      gsl::not_null<Scalar<DataVector>*> temp_gamma1,
      gsl::not_null<Scalar<DataVector>*> temp_gamma2,
      gsl::not_null<tnsr::a<DataVector, Dim>*> temp_gauge_function,
      gsl::not_null<tnsr::ab<DataVector, Dim>*>
          temp_spacetime_deriv_gauge_function,
      gsl::not_null<Scalar<DataVector>*> gamma1gamma2,
      gsl::not_null<Scalar<DataVector>*> pi_two_normals,
      gsl::not_null<Scalar<DataVector>*> normal_dot_gauge_constraint,
      gsl::not_null<Scalar<DataVector>*> gamma1_plus_1,
      gsl::not_null<tnsr::a<DataVector, Dim>*> pi_one_normal,
      gsl::not_null<tnsr::a<DataVector, Dim>*> gauge_constraint,
      gsl::not_null<tnsr::i<DataVector, Dim>*> phi_two_normals,
      gsl::not_null<tnsr::aa<DataVector, Dim>*>
          shift_dot_three_index_constraint,
      gsl::not_null<tnsr::ia<DataVector, Dim>*> phi_one_normal,
      gsl::not_null<tnsr::aB<DataVector, Dim>*> pi_2_up,
      gsl::not_null<tnsr::iaa<DataVector, Dim>*> three_index_constraint,
      gsl::not_null<tnsr::Iaa<DataVector, Dim>*> phi_1_up,
      gsl::not_null<tnsr::iaB<DataVector, Dim>*> phi_3_up,
      gsl::not_null<tnsr::abC<DataVector, Dim>*> christoffel_first_kind_3_up,
      gsl::not_null<Scalar<DataVector>*> lapse,
      gsl::not_null<tnsr::I<DataVector, Dim>*> shift,
      gsl::not_null<tnsr::ii<DataVector, Dim>*> spatial_metric,
      gsl::not_null<tnsr::II<DataVector, Dim>*> inverse_spatial_metric,
      gsl::not_null<Scalar<DataVector>*> det_spatial_metric,
      gsl::not_null<tnsr::AA<DataVector, Dim>*> inverse_spacetime_metric,
      gsl::not_null<tnsr::abb<DataVector, Dim>*> christoffel_first_kind,
      gsl::not_null<tnsr::Abb<DataVector, Dim>*> christoffel_second_kind,
      gsl::not_null<tnsr::a<DataVector, Dim>*> trace_christoffel,
      gsl::not_null<tnsr::A<DataVector, Dim>*> normal_spacetime_vector,
      gsl::not_null<tnsr::a<DataVector, Dim>*> normal_spacetime_one_form,
      gsl::not_null<tnsr::abb<DataVector, Dim>*> da_spacetime_metric,
      const tnsr::iaa<DataVector, Dim>& d_spacetime_metric,
      const tnsr::iaa<DataVector, Dim>& d_pi,
      const tnsr::ijaa<DataVector, Dim>& d_phi,
      const tnsr::aa<DataVector, Dim>& spacetime_metric,
      const tnsr::aa<DataVector, Dim>& pi,
      const tnsr::iaa<DataVector, Dim>& phi, const Scalar<DataVector>& gamma0,
      const Scalar<DataVector>& gamma1, const Scalar<DataVector>& gamma2,
      const tnsr::a<DataVector, Dim>& gauge_function,
      const tnsr::ab<DataVector, Dim>& spacetime_deriv_gauge_function) noexcept;
};
}  // namespace GeneralizedHarmonic
