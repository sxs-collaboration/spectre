// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/FaceNormal.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/Tags.hpp"
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
/// @{
/*!
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
std::array<DataVector, 4> characteristic_speeds(
    const Scalar<DataVector>& gamma_1, const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame>& shift,
    const tnsr::i<DataVector, Dim, Frame>& unit_normal_one_form);

template <size_t Dim, typename Frame>
void characteristic_speeds(
    gsl::not_null<std::array<DataVector, 4>*> char_speeds,
    const Scalar<DataVector>& gamma_1, const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame>& shift,
    const tnsr::i<DataVector, Dim, Frame>& unit_normal_one_form);

template <size_t Dim, typename Frame>
struct CharacteristicSpeedsCompute : Tags::CharacteristicSpeeds<Dim, Frame>,
                                     db::ComputeTag {
  using base = Tags::CharacteristicSpeeds<Dim, Frame>;
  using type = typename base::type;
  using argument_tags = tmpl::list<
      ::GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma1,
      gr::Tags::Lapse<DataVector>, gr::Tags::Shift<Dim, Frame, DataVector>,
      ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim, Frame>>>;

  using return_type = typename base::type;

  static void function(
      const gsl::not_null<return_type*> result,
      const Scalar<DataVector>& gamma_1, const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, Dim, Frame>& shift,
      const tnsr::i<DataVector, Dim, Frame>& unit_normal_one_form) {
    characteristic_speeds(result, gamma_1, lapse, shift, unit_normal_one_form);
  };
};
/// @}

/// @{
/*!
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
typename Tags::CharacteristicFields<Dim, Frame>::type characteristic_fields(
    const Scalar<DataVector>& gamma_2,
    const tnsr::II<DataVector, Dim, Frame>& inverse_spatial_metric,
    const tnsr::aa<DataVector, Dim, Frame>& spacetime_metric,
    const tnsr::aa<DataVector, Dim, Frame>& pi,
    const tnsr::iaa<DataVector, Dim, Frame>& phi,
    const tnsr::i<DataVector, Dim, Frame>& unit_normal_one_form);

template <size_t Dim, typename Frame>
void characteristic_fields(
    gsl::not_null<typename Tags::CharacteristicFields<Dim, Frame>::type*>
        char_fields,
    const Scalar<DataVector>& gamma_2,
    const tnsr::II<DataVector, Dim, Frame>& inverse_spatial_metric,
    const tnsr::aa<DataVector, Dim, Frame>& spacetime_metric,
    const tnsr::aa<DataVector, Dim, Frame>& pi,
    const tnsr::iaa<DataVector, Dim, Frame>& phi,
    const tnsr::i<DataVector, Dim, Frame>& unit_normal_one_form);

template <size_t Dim, typename Frame>
struct CharacteristicFieldsCompute : Tags::CharacteristicFields<Dim, Frame>,
                                     db::ComputeTag {
  using base = Tags::CharacteristicFields<Dim, Frame>;
  using return_type = typename base::type;
  using argument_tags = tmpl::list<
      ::GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma2,
      gr::Tags::InverseSpatialMetric<Dim, Frame, DataVector>,
      gr::Tags::SpacetimeMetric<Dim, Frame, DataVector>, Tags::Pi<Dim, Frame>,
      Tags::Phi<Dim, Frame>,
      ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim, Frame>>>;

  static constexpr auto function = static_cast<void (*)(
      const gsl::not_null<return_type*>, const Scalar<DataVector>&,
      const tnsr::II<DataVector, Dim, Frame>&,
      const tnsr::aa<DataVector, Dim, Frame>&,
      const tnsr::aa<DataVector, Dim, Frame>&,
      const tnsr::iaa<DataVector, Dim, Frame>&,
      const tnsr::i<DataVector, Dim, Frame>&)>(&characteristic_fields);
};
/// @}

/// @{
/*!
 * \brief For expressions used here to compute evolved fields from
 * characteristic ones, see \ref CharacteristicFieldsCompute.
 */
template <size_t Dim, typename Frame>
typename Tags::EvolvedFieldsFromCharacteristicFields<Dim, Frame>::type
evolved_fields_from_characteristic_fields(
    const Scalar<DataVector>& gamma_2,
    const tnsr::aa<DataVector, Dim, Frame>& u_psi,
    const tnsr::iaa<DataVector, Dim, Frame>& u_zero,
    const tnsr::aa<DataVector, Dim, Frame>& u_plus,
    const tnsr::aa<DataVector, Dim, Frame>& u_minus,
    const tnsr::i<DataVector, Dim, Frame>& unit_normal_one_form);

template <size_t Dim, typename Frame>
void evolved_fields_from_characteristic_fields(
    gsl::not_null<
        typename Tags::EvolvedFieldsFromCharacteristicFields<Dim, Frame>::type*>
        evolved_fields,
    const Scalar<DataVector>& gamma_2,
    const tnsr::aa<DataVector, Dim, Frame>& u_psi,
    const tnsr::iaa<DataVector, Dim, Frame>& u_zero,
    const tnsr::aa<DataVector, Dim, Frame>& u_plus,
    const tnsr::aa<DataVector, Dim, Frame>& u_minus,
    const tnsr::i<DataVector, Dim, Frame>& unit_normal_one_form);

template <size_t Dim, typename Frame>
struct EvolvedFieldsFromCharacteristicFieldsCompute
    : Tags::EvolvedFieldsFromCharacteristicFields<Dim, Frame>,
      db::ComputeTag {
  using base = Tags::EvolvedFieldsFromCharacteristicFields<Dim, Frame>;
  using return_type = typename base::type;
  using argument_tags = tmpl::list<
      GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma2,
      Tags::VSpacetimeMetric<Dim, Frame>, Tags::VZero<Dim, Frame>,
      Tags::VPlus<Dim, Frame>, Tags::VMinus<Dim, Frame>,
      ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim, Frame>>>;

  static constexpr auto function = static_cast<void (*)(
      const gsl::not_null<return_type*>, const Scalar<DataVector>& gamma_2,
      const tnsr::aa<DataVector, Dim, Frame>& u_psi,
      const tnsr::iaa<DataVector, Dim, Frame>& u_zero,
      const tnsr::aa<DataVector, Dim, Frame>& u_plus,
      const tnsr::aa<DataVector, Dim, Frame>& u_minus,
      const tnsr::i<DataVector, Dim, Frame>& unit_normal_one_form)>(
      &evolved_fields_from_characteristic_fields);
};
/// @}

namespace Tags{
struct LargestCharacteristicSpeed : db::SimpleTag {
  using type = double;
};
/*!
 * \brief Computes the largest magnitude of the characteristic speeds.
 */
template <size_t Dim, typename Frame>
struct ComputeLargestCharacteristicSpeed : db::ComputeTag,
                                           LargestCharacteristicSpeed {
  using argument_tags = tmpl::list<
      ::GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma1,
      gr::Tags::Lapse<DataVector>, gr::Tags::Shift<Dim, Frame, DataVector>,
      gr::Tags::SpatialMetric<Dim, Frame, DataVector>>;
  using return_type = double;
  using base = LargestCharacteristicSpeed;
  static void function(const gsl::not_null<double*> speed,
                       const Scalar<DataVector>& gamma_1,
                       const Scalar<DataVector>& lapse,
                       const tnsr::I<DataVector, Dim, Frame>& shift,
                       const tnsr::ii<DataVector, Dim, Frame>& spatial_metric);
};
}  // namespace Tags
}  // namespace GeneralizedHarmonic
