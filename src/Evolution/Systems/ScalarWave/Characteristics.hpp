// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/FaceNormal.hpp"
#include "Evolution/Systems/ScalarWave/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <typename>
class Variables;

namespace Tags {
template <typename Tag>
struct Normalized;
}  // namespace Tags
/// \endcond

namespace ScalarWave {
// @{
/*!
 * \ingroup ScalarWave
 * \brief Compute the characteristic speeds for the scalar wave system.
 *
 * Computes the speeds as described in "Optimal constraint projection for
 * hyperbolic evolution systems" by Holst et al. \cite Holst2004wt
 * [see text following Eq.(32)]. The characteristic fields' names used here
 * differ from this paper:
 *
 * \f{align*}
 * \mathrm{SpECTRE} && \mathrm{Holst} \\
 * v^{\hat \psi} && Z^1 \\
 * v^{\hat 0}_{i} && Z^{2}_{i} \\
 * v^{\hat \pm} && u^{1\pm}
 * \f}
 *
 * The corresponding characteristic speeds \f$\lambda_{\hat \alpha}\f$ are given
 * in the text following Eq.(38) of \cite Holst2004wt :
 *
 * \f{align*}
 * \lambda_{\hat \psi} =& 0 \\
 * \lambda_{\hat 0} =& 0 \\
 * \lambda_{\hat \pm} =& \pm 1.
 * \f}
 */
template <size_t Dim>
std::array<DataVector, 4> characteristic_speeds(
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        unit_normal_one_form) noexcept;

template <size_t Dim>
void characteristic_speeds(
    gsl::not_null<std::array<DataVector, 4>*> char_speeds,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        unit_normal_one_form) noexcept;

template <size_t Dim>
struct CharacteristicSpeedsCompute : Tags::CharacteristicSpeeds<Dim>,
                                     db::ComputeTag {
  using base = Tags::CharacteristicSpeeds<Dim>;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim>>>;

  static void function(gsl::not_null<return_type*> char_speeds,
                       const tnsr::i<DataVector, Dim, Frame::Inertial>&
                           unit_normal_one_form) noexcept {
    characteristic_speeds(char_speeds, unit_normal_one_form);
  }
};
// @}

// @{
/*!
 * \ingroup ScalarWave
 * \brief Computes characteristic fields from evolved fields
 *
 * \ref CharacteristicFieldsCompute and
 * \ref EvolvedFieldsFromCharacteristicFieldsCompute convert between
 * characteristic and evolved fields for the scalar-wave system.
 *
 * \ref CharacteristicFieldsCompute computes
 * characteristic fields as described in "Optimal constraint projection for
 * hyperbolic evolution systems" by Holst et al. \cite Holst2004wt .
 * Their names used here differ from this paper:
 *
 * \f{align*}
 * \mathrm{SpECTRE} && \mathrm{Holst} \\
 * v^{\hat \psi} && Z^1 \\
 * v^{\hat 0}_{i} && Z^{2}_{i} \\
 * v^{\hat \pm} && u^{1\pm}
 * \f}
 *
 * The characteristic fields \f${v}^{\hat \alpha}\f$ are given in terms of
 * the evolved fields by Eq.(33) - (35) of \cite Holst2004wt, respectively:
 *
 * \f{align*}
 * v^{\hat \psi} =& \psi \\
 * v^{\hat 0}_{i} =& (\delta^k_i - n_i n^k) \Phi_{k} := P^k_i \Phi_{k} \\
 * v^{\hat \pm} =& \Pi \pm n^i \Phi_{i} - \gamma_2\psi
 * \f}
 *
 * where \f$\psi\f$ is the scalar field, \f$\Phi_{i}=\partial_i \psi\f$ is an
 * auxiliary variable, \f$\Pi\f$ is a conjugate momentum, \f$\gamma_2\f$
 * is a constraint damping parameter, and \f$n_k\f$ is the unit normal to the
 * surface along which the characteristic fields are defined.
 *
 * \ref EvolvedFieldsFromCharacteristicFieldsCompute computes evolved fields
 * \f$u_\alpha\f$ in terms of the characteristic fields. This uses the inverse
 * of above relations:
 *
 * \f{align*}
 * \psi =& v^{\hat \psi}, \\
 * \Pi =& \frac{1}{2}(v^{\hat +} + v^{\hat -}) + \gamma_2 v^{\hat \psi}, \\
 * \Phi_{i} =& \frac{1}{2}(v^{\hat +} - v^{\hat -}) n_i + v^{\hat 0}_{i}.
 * \f}
 *
 * The corresponding characteristic speeds \f$\lambda_{\hat \alpha}\f$
 * are computed by \ref CharacteristicSpeedsCompute .
 */
template <size_t Dim>
Variables<tmpl::list<Tags::VPsi, Tags::VZero<Dim>, Tags::VPlus, Tags::VMinus>>
characteristic_fields(const Scalar<DataVector>& gamma_2,
                      const Scalar<DataVector>& psi,
                      const Scalar<DataVector>& pi,
                      const tnsr::i<DataVector, Dim, Frame::Inertial>& phi,
                      const tnsr::i<DataVector, Dim, Frame::Inertial>&
                          unit_normal_one_form) noexcept;

template <size_t Dim>
void characteristic_fields(
    gsl::not_null<Variables<
        tmpl::list<Tags::VPsi, Tags::VZero<Dim>, Tags::VPlus, Tags::VMinus>>*>
        char_fields,
    const Scalar<DataVector>& gamma_2, const Scalar<DataVector>& psi,
    const Scalar<DataVector>& pi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& phi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        unit_normal_one_form) noexcept;

template <size_t Dim>
struct CharacteristicFieldsCompute : Tags::CharacteristicFields<Dim>,
                                     db::ComputeTag {
  using base = Tags::CharacteristicFields<Dim>;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<Tags::ConstraintGamma2, Psi, Pi, Phi<Dim>,
                 ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim>>>;

  static void function(const gsl::not_null<return_type*> char_fields,
                       const Scalar<DataVector>& gamma_2,
                       const Scalar<DataVector>& psi,
                       const Scalar<DataVector>& pi,
                       const tnsr::i<DataVector, Dim, Frame::Inertial>& phi,
                       const tnsr::i<DataVector, Dim, Frame::Inertial>&
                           unit_normal_one_form) noexcept {
    characteristic_fields(char_fields, gamma_2, psi, pi, phi,
                          unit_normal_one_form);
  };
};
// @}

// @{
/*!
 * \ingroup ScalarWave
 * \brief Compute evolved fields from characteristic fields.
 *
 * For expressions used here to compute evolved fields from characteristic ones,
 * see \ref CharacteristicFieldsCompute.
 */
template <size_t Dim>
Variables<tmpl::list<Psi, Pi, Phi<Dim>>>
evolved_fields_from_characteristic_fields(
    const Scalar<DataVector>& gamma_2, const Scalar<DataVector>& v_psi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& v_zero,
    const Scalar<DataVector>& v_plus, const Scalar<DataVector>& v_minus,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        unit_normal_one_form) noexcept;

template <size_t Dim>
void evolved_fields_from_characteristic_fields(
    gsl::not_null<Variables<tmpl::list<Psi, Pi, Phi<Dim>>>*> evolved_fields,
    const Scalar<DataVector>& gamma_2, const Scalar<DataVector>& v_psi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& v_zero,
    const Scalar<DataVector>& v_plus, const Scalar<DataVector>& v_minus,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        unit_normal_one_form) noexcept;

template <size_t Dim>
struct EvolvedFieldsFromCharacteristicFieldsCompute
    : Tags::EvolvedFieldsFromCharacteristicFields<Dim>,
      db::ComputeTag {
  using base = Tags::EvolvedFieldsFromCharacteristicFields<Dim>;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<Tags::ConstraintGamma2, Tags::VPsi, Tags::VZero<Dim>,
                 Tags::VPlus, Tags::VMinus,
                 ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim>>>;

  static void function(const gsl::not_null<return_type*> evolved_fields,
                       const Scalar<DataVector>& gamma_2,
                       const Scalar<DataVector>& v_psi,
                       const tnsr::i<DataVector, Dim, Frame::Inertial>& v_zero,
                       const Scalar<DataVector>& v_plus,
                       const Scalar<DataVector>& v_minus,
                       const tnsr::i<DataVector, Dim, Frame::Inertial>&
                           unit_normal_one_form) noexcept {
    evolved_fields_from_characteristic_fields(evolved_fields, gamma_2, v_psi,
                                              v_zero, v_plus, v_minus,
                                              unit_normal_one_form);
  };
};
// @}
}  // namespace ScalarWave
