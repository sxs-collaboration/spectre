// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/FaceNormal.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace gsl {
template <class T>
class not_null;
}  // namespace gsl
namespace Tags {
template <typename Tag>
struct Normalized;
}  // namespace Tags
/// \endcond

// IWYU pragma: no_forward_declare Tensor

namespace GeneralizedHarmonic {
// @{
/*!
 * \ingroup GeneralizedHarmonic
 * \brief Compute the characteristic speeds for the generalized harmonic system.
 *
 * Computes the speeds as described in "A New Generalized Harmonic
 * Evolution System" by Lindblom et. al \cite Lindblom2005qh
 * [see text following Eq.(34)]. The characteristic fields' names used here
 * differ from this paper:
 *
 * \f{align*}
 * \mathrm{SpECTRE} && \mathrm{Lindblom} \\
 * u^{\psi}_{ab} && u^\hat{0}_{ab} \\
 * u^0_{iab} && u^\hat{2}_{iab} \\
 * u^{\pm}_{ab} && u^{\hat{1}\pm}_{ab}
 * \f}
 *
 * The corresponding characteristic speeds \f$v\f$ are given in the text between
 * Eq.(34) and Eq.(35) of \cite Lindblom2005qh :
 *
 * \f{align*}
 * v_{\psi} =& -(1 + \gamma_1) n_k N^k \\
 * v_{0} =& -n_k N^k \\
 * v_{\pm} =& -n_k N^k \pm N
 * \f}
 *
 * where \f$N, N^k\f$ are the lapse and shift respectively, \f$\gamma_1\f$ is a
 * constraint damping parameter, and \f$n_k\f$ is the unit normal to the
 * surface.
 */
template <size_t Dim, typename Frame>
struct CharacteristicSpeedsCompute : Tags::CharacteristicSpeeds<Dim, Frame>,
                                     db::ComputeTag {
  using base = Tags::CharacteristicSpeeds<Dim, Frame>;
  using type = typename base::type;
  using argument_tags = tmpl::list<
      Tags::ConstraintGamma1, gr::Tags::Lapse<DataVector>,
      gr::Tags::Shift<Dim, Frame, DataVector>,
      ::Tags::Normalized<::Tags::UnnormalizedFaceNormal<Dim, Frame>>>;

  static typename Tags::CharacteristicSpeeds<Dim, Frame>::type function(
      const Scalar<DataVector>& gamma_1, const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, Dim, Frame>& shift,
      const tnsr::i<DataVector, Dim, Frame>& unit_normal_one_form) noexcept;
};

template <size_t Dim, typename Frame>
void compute_characteristic_speeds(
    gsl::not_null<typename Tags::CharacteristicSpeeds<Dim, Frame>::type*>
        char_speeds,
    const Scalar<DataVector>& gamma_1, const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame>& shift,
    const tnsr::i<DataVector, Dim, Frame>& unit_normal_one_form) noexcept;
// @}

// @{
/*!
 * \ingroup GeneralizedHarmonic
 * \brief Computes characteristic fields from evolved fields
 *
 * \ref CharacteristicFieldsCompute and
 * \ref EvolvedFieldsFromCharacteristicFieldsCompute convert between
 * characteristic and evolved fields for the generalized harmonic system.
 *
 * \ref CharacteristicFieldsCompute computes
 * characteristic fields as described in "A New Generalized Harmonic
 * Evolution System" by Lindblom et. al \cite Lindblom2005qh .
 * Their names used here differ from this paper:
 *
 * \f{align*}
 * \mathrm{SpECTRE} && \mathrm{Lindblom} \\
 * u^{\psi}_{ab} && u^\hat{0}_{ab} \\
 * u^0_{iab} && u^\hat{2}_{iab} \\
 * u^{\pm}_{ab} && u^{\hat{1}\pm}_{ab}
 * \f}
 *
 * The characteristic fields \f$u\f$ are given in terms of the evolved fields by
 * Eq.(32) - (34) of \cite Lindblom2005qh, respectively:
 * \f{align*}
 * u^{\psi}_{ab} =& \psi_{ab} \\
 * u^0_{iab} =& (\delta^k_i - n_i n^k) \Phi_{kab} := P^k_i \Phi_{kab} \\
 * u^{\pm}_{ab} =& \Pi_{ab} \pm n^i \Phi_{iab} - \gamma_2\psi_{ab}
 * \f}
 *
 * where \f$\psi_{ab}\f$ is the spacetime metric, \f$\Pi_{ab}\f$ and
 * \f$\Phi_{iab}\f$ are evolved generalized harmonic fields introduced by first
 * derivatives of \f$\psi_{ab}\f$, \f$\gamma_2\f$ is a constraint damping
 * parameter, and \f$n_k\f$ is the unit normal to the surface.
 *
 * \ref EvolvedFieldsFromCharacteristicFieldsCompute computes evolved fields
 * \f$w\f$ in terms of the characteristic fields. This uses the inverse of
 * above relations:
 *
 * \f{align*}
 * \psi_{ab} =& u^{\psi}_{ab}, \\
 * \Pi_{ab} =& \frac{1}{2}(u^{+}_{ab} + u^{-}_{ab}) + \gamma_2 u^{\psi}_{ab}, \\
 * \Phi_{iab} =& \frac{1}{2}(u^{+}_{ab} - u^{-}_{ab}) n_i + u^0_{iab}.
 * \f}
 *
 * The corresponding characteristic speeds \f$v\f$ are computed by
 * \ref CharacteristicSpeedsCompute .
 */
template <size_t Dim, typename Frame>
struct CharacteristicFieldsCompute : Tags::CharacteristicFields<Dim, Frame>,
                                     db::ComputeTag {
  using base = Tags::CharacteristicFields<Dim, Frame>;
  using type = typename base::type;
  using argument_tags = tmpl::list<
      Tags::ConstraintGamma2,
      gr::Tags::InverseSpatialMetric<Dim, Frame, DataVector>,
      gr::Tags::SpacetimeMetric<Dim, Frame, DataVector>, Tags::Pi<Dim, Frame>,
      Tags::Phi<Dim, Frame>,
      ::Tags::Normalized<::Tags::UnnormalizedFaceNormal<Dim, Frame>>>;

  static typename Tags::CharacteristicFields<Dim, Frame>::type function(
      const Scalar<DataVector>& gamma_2,
      const tnsr::II<DataVector, Dim, Frame>& inverse_spatial_metric,
      const tnsr::aa<DataVector, Dim, Frame>& spacetime_metric,
      const tnsr::aa<DataVector, Dim, Frame>& pi,
      const tnsr::iaa<DataVector, Dim, Frame>& phi,
      const tnsr::i<DataVector, Dim, Frame>& unit_normal_one_form) noexcept;
};

template <size_t Dim, typename Frame>
void compute_characteristic_fields(
    gsl::not_null<typename Tags::CharacteristicFields<Dim, Frame>::type*>
        char_fields,
    const Scalar<DataVector>& gamma_2,
    const tnsr::II<DataVector, Dim, Frame>& inverse_spatial_metric,
    const tnsr::aa<DataVector, Dim, Frame>& spacetime_metric,
    const tnsr::aa<DataVector, Dim, Frame>& pi,
    const tnsr::iaa<DataVector, Dim, Frame>& phi,
    const tnsr::i<DataVector, Dim, Frame>& unit_normal_one_form) noexcept;
// @}

// @{
/*!
 * \ingroup GeneralizedHarmonic
 * \brief For expressions used here to compute evolved fields from
 * characteristic ones, see \ref CharacteristicFieldsCompute.
 */
template <size_t Dim, typename Frame>
struct EvolvedFieldsFromCharacteristicFieldsCompute
    : Tags::EvolvedFieldsFromCharacteristicFields<Dim, Frame>,
      db::ComputeTag {
  using base = Tags::EvolvedFieldsFromCharacteristicFields<Dim, Frame>;
  using type = typename base::type;
  using argument_tags = tmpl::list<
      Tags::ConstraintGamma2, Tags::UPsi<Dim, Frame>, Tags::UZero<Dim, Frame>,
      Tags::UPlus<Dim, Frame>, Tags::UMinus<Dim, Frame>,
      ::Tags::Normalized<::Tags::UnnormalizedFaceNormal<Dim, Frame>>>;

  static typename Tags::EvolvedFieldsFromCharacteristicFields<Dim, Frame>::type
  function(
      const Scalar<DataVector>& gamma_2,
      const tnsr::aa<DataVector, Dim, Frame>& u_psi,
      const tnsr::iaa<DataVector, Dim, Frame>& u_zero,
      const tnsr::aa<DataVector, Dim, Frame>& u_plus,
      const tnsr::aa<DataVector, Dim, Frame>& u_minus,
      const tnsr::i<DataVector, Dim, Frame>& unit_normal_one_form) noexcept;
};

template <size_t Dim, typename Frame>
void compute_evolved_fields_from_characteristic_fields(
    gsl::not_null<
        typename Tags::EvolvedFieldsFromCharacteristicFields<Dim, Frame>::type*>
        evolved_fields,
    const Scalar<DataVector>& gamma_2,
    const tnsr::aa<DataVector, Dim, Frame>& u_psi,
    const tnsr::iaa<DataVector, Dim, Frame>& u_zero,
    const tnsr::aa<DataVector, Dim, Frame>& u_plus,
    const tnsr::aa<DataVector, Dim, Frame>& u_minus,
    const tnsr::i<DataVector, Dim, Frame>& unit_normal_one_form) noexcept;

// @}

/*!
 * \ingroup GeneralizedHarmonic
 * \brief Computes the largest magnitude of the characteristic speeds.
 */
template <size_t Dim, typename Frame>
struct ComputeLargestCharacteristicSpeed {
  using argument_tags = tmpl::list<Tags::CharacteristicSpeeds<Dim, Frame>>;
  static double apply(const std::array<DataVector, 4>& char_speeds) noexcept;
};
}  // namespace GeneralizedHarmonic
