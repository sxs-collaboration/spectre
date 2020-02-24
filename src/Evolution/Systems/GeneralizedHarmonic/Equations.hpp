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
  using argument_tags = tmpl::list<
      gr::Tags::SpacetimeMetric<Dim>, Tags::Pi<Dim>, Tags::Phi<Dim>,
      Tags::ConstraintGamma1, Tags::ConstraintGamma2, gr::Tags::Lapse<>,
      gr::Tags::Shift<Dim>, gr::Tags::InverseSpatialMetric<Dim>,
      ::Tags::Normalized<
          domain::Tags::UnnormalizedFaceNormal<Dim, Frame::Inertial>>>;

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

/*!
 * \ingroup NumericalFluxesGroup
 * \brief Computes the generalized-harmonic upwind flux
 *
 * The upwind flux in general is given by Eq. (6.3) of \cite Teukolsky2015ega :
 * \f{eqnarray}{
 * D = S \Lambda^{\rm pos} S^{-1} u^{\rm int}
 *   + S \Lambda^{\rm neg} S^{-1} u^{\rm ext},
 * \f}
 * where \f$u\f$ is a vector of the evolved variables, \f$S^{-1}\f$
 * maps the evolved variables into the characteristic variables \f$S^{-1}u\f$,
 * \f$S\f$ maps the characteristic variables into the evolved variables,
 * and \f$\Lambda\f$ is a diagonal matrix of the average characteristic
 * speed at the interface.
 *
 * Here, \f$S^{-1}u^{\rm int}\f$ represents the
 * characteristic variables at the element interface, computed using
 * evolved variables from the element interior;
 * \f$S^{-1}u^{\rm ext}\f$ represents the characteristic variables at the
 * element interface, computed using evolved variables from the exterior,
 * neighboring element; \f$\Lambda^{\rm pos}\f$ is a diagonal
 * matrix whose nonzero entries are the average characteristic speeds
 * that are positive ("outgoing", i.e. leaving the element); and
 * \f$\Lambda^{\rm neg}\f$ is a diagonal matrix whose nonzero entries are
 * the average characteristic speeds that are negative ("incoming", i.e.
 * entering the element). If a characteristic field
 * \f$U^{\rm int}\f$ has a characteristic speed \f$v^{\rm int}\f$ and the same
 * field in the exterior \f$U^{\rm ext}\f$ has speed \f$v^{\rm ext}\f$, then
 * the average characteristic speed is
 * \f$v^{\rm avg} = (1/2)(v^{\rm int} + v^{\rm ext})\f$.
 *
 * This function implements the upwind flux for the generalized harmonic
 * system. First, it computes the characteristic variables using i) the
 * evolved variables from the interior and ii) the evolved variables from the
 * exterior. Then, it computes \f$\Lambda^{\rm pos}\f$ and
 * \f$\Lambda^{\rm neg}\f$ from the average characteristic speeds.
 * Then, it computes the combination
 * \f$\Lambda^{\rm pos} S^{-1} u^{\rm int}
 * + \Lambda^{\rm neg} S^{-1} u^{\rm ext}\f$. Finally, it
 * applies \f$S\f$ by converting the result back from characteristic to
 * evolved variables, using the unit normal vector of the element and
 * the average value of the field
 * \f$\gamma_2^{\rm avg} = (1/2)(\gamma_2^{\rm int} + \gamma_2^{\rm ext})\f$,
 * where here \f$\gamma_2^{\rm int}\f$ is the value of \f$\gamma_2\f$
 * in the interior and \f$\gamma_2^{\rm ext}\f$ is the value of
 * \f$\gamma_2\f$ in the exterior.
 *
 * \warning as currently implemented, this flux does not satisfy the
 * generalized Rankine-Hugoniot condition
 * \f$D^{\rm int}([S^{-1} u]^{\rm int}, [S^{-1} u]^{\rm ext})
 * = - D^{\rm ext}([S^{-1} u]^{\rm ext}, [S^{-1} u]^{\rm int})\f$,
 * which enforces that the flux leaving the interior is the flux entering
 * the exterior and vice versa. This condition is necessary for a
 * well-balanced scheme. Use this flux with caution; so far it has
 * been tested for evolving Kerr-Schild and a perturbed Kerr black hole.
 */
template <size_t Dim>
struct UpwindFlux {
 public:
  using options = tmpl::list<>;
  static constexpr OptionString help = {
      "Computes the generalized harmonic upwind flux."};

  // clang-tidy: non-const reference
  void pup(PUP::er& /*p*/) noexcept {}  // NOLINT

  // This is the data needed to compute the numerical flux.
  // `dg::SendBoundaryFluxes` calls `package_data` to store these tags in a
  // Variables. Local and remote values of this data are then combined in the
  // `()` operator.
  using package_tags = tmpl::list<
      gr::Tags::SpacetimeMetric<Dim, Frame::Inertial, DataVector>,
      Tags::Pi<Dim, Frame::Inertial>, Tags::Phi<Dim, Frame::Inertial>,
      gr::Tags::Lapse<DataVector>,
      gr::Tags::Shift<Dim, Frame::Inertial, DataVector>,
      gr::Tags::InverseSpatialMetric<Dim, Frame::Inertial, DataVector>,
      Tags::ConstraintGamma1, Tags::ConstraintGamma2,
      ::Tags::Normalized<
          domain::Tags::UnnormalizedFaceNormal<Dim, Frame::Inertial>>>;

  // These tags on the interface of the element are passed to
  // `package_data` to provide the data needed to compute the numerical fluxes.
  using argument_tags = tmpl::list<
      gr::Tags::SpacetimeMetric<Dim, Frame::Inertial, DataVector>,
      Tags::Pi<Dim, Frame::Inertial>, Tags::Phi<Dim, Frame::Inertial>,
      gr::Tags::Lapse<DataVector>,
      gr::Tags::Shift<Dim, Frame::Inertial, DataVector>,
      gr::Tags::InverseSpatialMetric<Dim, Frame::Inertial, DataVector>,
      Tags::ConstraintGamma1, Tags::ConstraintGamma2,
      ::Tags::Normalized<
          domain::Tags::UnnormalizedFaceNormal<Dim, Frame::Inertial>>>;

  // pseudo-interface: used internally by Algorithm infrastructure, not
  // user-level code
  // Following the not-null pointer to packaged_data, this function expects as
  // arguments the databox types of the `argument_tags`.
  void package_data(
      gsl::not_null<Variables<package_tags>*> packaged_data,
      const tnsr::aa<DataVector, Dim, Frame::Inertial>& spacetime_metric,
      const tnsr::aa<DataVector, Dim, Frame::Inertial>& pi,
      const tnsr::iaa<DataVector, Dim, Frame::Inertial>& phi,
      const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& shift,
      const tnsr::II<DataVector, Dim, Frame::Inertial>& inverse_spatial_metric,
      const Scalar<DataVector>& gamma1, const Scalar<DataVector>& gamma2,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& interface_unit_normal)
      const noexcept;

  // pseudo-interface: used internally by Algorithm infrastructure, not
  // user-level code
  // The arguments are first the system::variables_tag::tags_list wrapped in
  // Tags::NormalDotNumericalFlux as not-null pointers to write the results
  // into, then the package_tags on the interior side of the mortar followed by
  // the package_tags on the exterior side.
  void operator()(
      gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*>
          psi_normal_dot_numerical_flux,
      gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*>
          pi_normal_dot_numerical_flux,
      gsl::not_null<tnsr::iaa<DataVector, Dim, Frame::Inertial>*>
          phi_normal_dot_numerical_flux,
      const tnsr::aa<DataVector, Dim, Frame::Inertial>& spacetime_metric_int,
      const tnsr::aa<DataVector, Dim, Frame::Inertial>& pi_int,
      const tnsr::iaa<DataVector, Dim, Frame::Inertial>& phi_int,
      const Scalar<DataVector>& lapse_int,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& shift_int,
      const tnsr::II<DataVector, Dim, Frame::Inertial>&
          inverse_spatial_metric_int,
      const Scalar<DataVector>& gamma1_int,
      const Scalar<DataVector>& gamma2_int,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          interface_unit_normal_int,
      const tnsr::aa<DataVector, Dim, Frame::Inertial>& spacetime_metric_ext,
      const tnsr::aa<DataVector, Dim, Frame::Inertial>& pi_ext,
      const tnsr::iaa<DataVector, Dim, Frame::Inertial>& phi_ext,
      const Scalar<DataVector>& lapse_ext,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& shift_ext,
      const tnsr::II<DataVector, Dim, Frame::Inertial>&
          inverse_spatial_metric_ext,
      const Scalar<DataVector>& gamma1_ext,
      const Scalar<DataVector>& gamma2_ext,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          interface_unit_normal_ext) const noexcept;
};
}  // namespace GeneralizedHarmonic
