// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class template GeneralizedHarmonicEquations.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/FaceNormal.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"  // IWYU pragma: keep
#include "Options/Options.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"  // IWYU pragma: keep
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;

namespace PUP {
class er;
}  // namespace PUP

namespace Tags {
template <typename Tag>
struct NormalDotNumericalFlux;
}  // namespace Tags

namespace gsl {
template <class T>
class not_null;
}  // namespace gsl

template <typename TagsList>
class Variables;

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
struct ComputeDuDt {
 public:
  template <template <class> class StepPrefix>
  using return_tags = tmpl::list<
      db::add_tag_prefix<StepPrefix, gr::Tags::SpacetimeMetric<
                                         Dim, Frame::Inertial, DataVector>>,
      db::add_tag_prefix<StepPrefix, Tags::Pi<Dim>>,
      db::add_tag_prefix<StepPrefix, Tags::Phi<Dim>>>;

  using argument_tags = tmpl::list<
      gr::Tags::SpacetimeMetric<Dim>, Tags::Pi<Dim>, Tags::Phi<Dim>,
      ::Tags::deriv<gr::Tags::SpacetimeMetric<Dim>, tmpl::size_t<Dim>,
                    Frame::Inertial>,
      ::Tags::deriv<Tags::Pi<Dim>, tmpl::size_t<Dim>, Frame::Inertial>,
      ::Tags::deriv<Tags::Phi<Dim>, tmpl::size_t<Dim>, Frame::Inertial>,
      Tags::ConstraintGamma0, Tags::ConstraintGamma1, Tags::ConstraintGamma2,
      Tags::GaugeH<Dim>, Tags::SpacetimeDerivGaugeH<Dim>>;

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
      const tnsr::ab<DataVector, Dim>& spacetime_deriv_gauge_function);
};

/*!
 * \brief Set the normal dot the flux to zero since the generalized harmonic
 * system has no fluxes and they're currently still needed for the evolution
 * scheme.
 */
template <size_t Dim>
struct ComputeNormalDotFluxes {
 public:
  using argument_tags = tmpl::list<gr::Tags::SpacetimeMetric<Dim>>;

  static void apply(
      gsl::not_null<tnsr::aa<DataVector, Dim>*>
          spacetime_metric_normal_dot_flux,
      gsl::not_null<tnsr::aa<DataVector, Dim>*> pi_normal_dot_flux,
      gsl::not_null<tnsr::iaa<DataVector, Dim>*> phi_normal_dot_flux,
      const tnsr::aa<DataVector, Dim>& spacetime_metric) noexcept;
};
}  // namespace GeneralizedHarmonic
