// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/FaceNormal.hpp"
#include "Evolution/Systems/CurvedScalarWave/Constraints.hpp"
#include "Evolution/Systems/CurvedScalarWave/TagsDeclarations.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "Utilities/TMPL.hpp"

namespace CurvedScalarWave {
// @{
/*!
 * \brief Compute the characteristic speeds for the scalar wave system in curved
 * spacetime.
 *
 * Computes the speeds as described in "Optimal constraint projection for
 * hyperbolic evolution systems" by Holst et. al \cite Holst2004wt
 * [see text following Eq. (32)]. The characteristic fields' names used here
 * are similar to the paper:
 *
 * \f{align*}
 * \mathrm{SpECTRE} && \mathrm{Holst} \\
 * v^{\hat \psi} && Z^1 \\
 * v^{\hat 0}_{i} && Z^{2}_{i} \\
 * v^{\hat \pm} && u^{1\pm}
 * \f}
 *
 * The corresponding characteristic speeds \f$\lambda\f$ are given in the text
 * following Eq. (38) of \cite Holst2004wt :
 *
 * \f{align*}
 * \lambda_{\hat \psi} =& -(1 + \gamma_1) n_k N^k \\
 * \lambda_{\hat 0} =& -n_k N^k \\
 * \lambda_{\hat \pm} =& -n_k N^k \pm N
 * \f}
 *
 * where \f$n_k\f$ is the unit normal to the surface.
 */
template <size_t SpatialDim>
std::array<DataVector, 4> characteristic_speeds(
    const Scalar<DataVector>& gamma_1, const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, SpatialDim, Frame::Inertial>& shift,
    const tnsr::i<DataVector, SpatialDim, Frame::Inertial>&
        unit_normal_one_form) noexcept;

template <size_t SpatialDim>
void characteristic_speeds(
    gsl::not_null<std::array<DataVector, 4>*> char_speeds,
    const Scalar<DataVector>& gamma_1, const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, SpatialDim, Frame::Inertial>& shift,
    const tnsr::i<DataVector, SpatialDim, Frame::Inertial>&
        unit_normal_one_form) noexcept;

template <size_t SpatialDim>
struct CharacteristicSpeedsCompute : Tags::CharacteristicSpeeds<SpatialDim>,
                                     db::ComputeTag {
  using base = Tags::CharacteristicSpeeds<SpatialDim>;
  using return_type = typename base::type;
  using argument_tags = tmpl::list<
      Tags::ConstraintGamma1, gr::Tags::Lapse<DataVector>,
      gr::Tags::Shift<SpatialDim, Frame::Inertial, DataVector>,
      ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<SpatialDim>>>;

  static constexpr void function(
      gsl::not_null<return_type*> result, const Scalar<DataVector>& gamma_1,
      const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, SpatialDim, Frame::Inertial>& shift,
      const tnsr::i<DataVector, SpatialDim, Frame::Inertial>&
          unit_normal_one_form) noexcept {
    characteristic_speeds<SpatialDim>(result, gamma_1, lapse, shift,
                                      unit_normal_one_form);
  }
};
// @}

// @{
/*!
 * \brief Computes characteristic fields from evolved fields
 *
 * \ref CharacteristicFieldsCompute and
 * \ref EvolvedFieldsFromCharacteristicFieldsCompute convert between
 * characteristic and evolved fields for the scalar-wave system in curved
 * spacetime.
 *
 * \ref CharacteristicFieldsCompute computes
 * characteristic fields as described in "Optimal constraint projection for
 * hyperbolic evolution systems" by Holst et. al \cite Holst2004wt .
 * Their names used here differ from this paper:
 *
 * \f{align*}
 * \mathrm{SpECTRE} && \mathrm{Holst} \\
 * v^{\hat \psi} && Z^1 \\
 * v^{\hat 0}_{i} && Z^{2}_{i} \\
 * v^{\hat \pm} && u^{1\pm}
 * \f}
 *
 * The characteristic fields \f$u\f$ are given in terms of the evolved fields by
 * Eq. (33) - (35) of \cite Holst2004wt, respectively:
 *
 * \f{align*}
 * v^{\hat \psi} =& \psi \\
 * v^{\hat 0}_{i} =& (\delta^k_i - n_i n^k) \Phi_{k} := P^k_i \Phi_{k} \\
 * v^{\hat \pm} =& \Pi \pm n^i \Phi_{i} - \gamma_2\psi
 * \f}
 *
 * where \f$\psi\f$ is the scalar field, \f$\Pi\f$ and \f$\Phi_{i}\f$ are
 * evolved fields introduced by first derivatives of \f$\psi\f$, \f$\gamma_2\f$
 * is a constraint damping parameter, and \f$n_k\f$ is the unit normal to the
 * surface.
 *
 * \ref EvolvedFieldsFromCharacteristicFieldsCompute computes evolved fields
 * \f$w\f$ in terms of the characteristic fields. This uses the inverse of
 * above relations (c.f. Eq. (36) - (38) of \cite Holst2004wt ):
 *
 * \f{align*}
 * \psi =& v^{\hat \psi}, \\
 * \Pi =& \frac{1}{2}(v^{\hat +} + v^{\hat -}) + \gamma_2 v^{\hat \psi}, \\
 * \Phi_{i} =& \frac{1}{2}(v^{\hat +} - v^{\hat -}) n_i + v^{\hat 0}_{i}.
 * \f}
 *
 * The corresponding characteristic speeds \f$\lambda\f$ are computed by
 * \ref CharacteristicSpeedsCompute .
 */
template <size_t SpatialDim>
Variables<
    tmpl::list<Tags::VPsi, Tags::VZero<SpatialDim>, Tags::VPlus, Tags::VMinus>>
characteristic_fields(
    const Scalar<DataVector>& gamma_2,
    const tnsr::II<DataVector, SpatialDim, Frame::Inertial>&
        inverse_spatial_metric,
    const Scalar<DataVector>& psi, const Scalar<DataVector>& pi,
    const tnsr::i<DataVector, SpatialDim, Frame::Inertial>& phi,
    const tnsr::i<DataVector, SpatialDim, Frame::Inertial>&
        unit_normal_one_form) noexcept;

template <size_t SpatialDim>
void characteristic_fields(
    gsl::not_null<Variables<tmpl::list<Tags::VPsi, Tags::VZero<SpatialDim>,
                                       Tags::VPlus, Tags::VMinus>>*>
        char_fields,
    const Scalar<DataVector>& gamma_2,
    const tnsr::II<DataVector, SpatialDim, Frame::Inertial>&
        inverse_spatial_metric,
    const Scalar<DataVector>& psi, const Scalar<DataVector>& pi,
    const tnsr::i<DataVector, SpatialDim, Frame::Inertial>& phi,
    const tnsr::i<DataVector, SpatialDim, Frame::Inertial>&
        unit_normal_one_form) noexcept;

template <size_t SpatialDim>
struct CharacteristicFieldsCompute : Tags::CharacteristicFields<SpatialDim>,
                                     db::ComputeTag {
  using base = Tags::CharacteristicFields<SpatialDim>;
  using return_type = typename base::type;
  using argument_tags = tmpl::list<
      Tags::ConstraintGamma2,
      gr::Tags::InverseSpatialMetric<SpatialDim, Frame::Inertial, DataVector>,
      Psi, Pi, Phi<SpatialDim>,
      ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<SpatialDim>>>;

  static constexpr void function(
      gsl::not_null<return_type*> result, const Scalar<DataVector>& gamma_2,
      const tnsr::II<DataVector, SpatialDim, Frame::Inertial>&
          inverse_spatial_metric,
      const Scalar<DataVector>& psi, const Scalar<DataVector>& pi,
      const tnsr::i<DataVector, SpatialDim, Frame::Inertial>& phi,
      const tnsr::i<DataVector, SpatialDim, Frame::Inertial>&
          unit_normal_one_form) noexcept {
    characteristic_fields<SpatialDim>(result, gamma_2, inverse_spatial_metric,
                                      psi, pi, phi, unit_normal_one_form);
  }
};
// @}

// @{
/*!
 * \brief For expressions used here to compute evolved fields from
 * characteristic ones, see \ref CharacteristicFieldsCompute.
 */
template <size_t SpatialDim>
Variables<tmpl::list<Psi, Pi, Phi<SpatialDim>>>
evolved_fields_from_characteristic_fields(
    const Scalar<DataVector>& gamma_2, const Scalar<DataVector>& v_psi,
    const tnsr::i<DataVector, SpatialDim, Frame::Inertial>& v_zero,
    const Scalar<DataVector>& v_plus, const Scalar<DataVector>& v_minus,
    const tnsr::i<DataVector, SpatialDim, Frame::Inertial>&
        unit_normal_one_form) noexcept;

template <size_t SpatialDim>
void evolved_fields_from_characteristic_fields(
    gsl::not_null<Variables<tmpl::list<Psi, Pi, Phi<SpatialDim>>>*>
        evolved_fields,
    const Scalar<DataVector>& gamma_2, const Scalar<DataVector>& v_psi,
    const tnsr::i<DataVector, SpatialDim, Frame::Inertial>& v_zero,
    const Scalar<DataVector>& v_plus, const Scalar<DataVector>& v_minus,
    const tnsr::i<DataVector, SpatialDim, Frame::Inertial>&
        unit_normal_one_form) noexcept;

template <size_t SpatialDim>
struct EvolvedFieldsFromCharacteristicFieldsCompute
    : Tags::EvolvedFieldsFromCharacteristicFields<SpatialDim>,
      db::ComputeTag {
  using base = Tags::EvolvedFieldsFromCharacteristicFields<SpatialDim>;
  using return_type = typename base::type;
  using argument_tags = tmpl::list<
      Tags::ConstraintGamma2, Tags::VPsi, Tags::VZero<SpatialDim>, Tags::VPlus,
      Tags::VMinus,
      ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<SpatialDim>>>;

  static constexpr void function(
      gsl::not_null<return_type*> result, const Scalar<DataVector>& gamma_2,
      const Scalar<DataVector>& v_psi,
      const tnsr::i<DataVector, SpatialDim, Frame::Inertial>& v_zero,
      const Scalar<DataVector>& v_plus, const Scalar<DataVector>& v_minus,
      const tnsr::i<DataVector, SpatialDim, Frame::Inertial>&
          unit_normal_one_form) noexcept {
    evolved_fields_from_characteristic_fields<SpatialDim>(
        result, gamma_2, v_psi, v_zero, v_plus, v_minus, unit_normal_one_form);
  }
};
// @}

/*!
 * \brief Computes the largest magnitude of the characteristic speeds.
 */
template <size_t SpatialDim>
struct ComputeLargestCharacteristicSpeed {
  using argument_tags = tmpl::list<Tags::CharacteristicSpeeds<SpatialDim>>;
  static double apply(const std::array<DataVector, 4>& char_speeds) noexcept;
};
}  // namespace CurvedScalarWave
