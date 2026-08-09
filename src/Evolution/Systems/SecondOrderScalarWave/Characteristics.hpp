// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/FaceNormal.hpp"
#include "Evolution/Systems/SecondOrderScalarWave/Tags.hpp"
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

namespace SecondOrderScalarWave {
/// @{
/*!
 * \brief Compute the characteristic speeds for the second-order scalar wave
 * system.
 *
 * The characteristic fields are, in order, \f$v^0_i\f$ (`Tags::VZero`),
 * \f$v^+\f$ (`Tags::VPlus`), and \f$v^-\f$ (`Tags::VMinus`). Their
 * characteristic speeds are 0, +1, and -1, respectively.
 */
template <size_t Dim>
std::array<DataVector, 3> characteristic_speeds(
    const tnsr::i<DataVector, Dim, Frame::Inertial>& unit_normal_one_form);

template <size_t Dim>
void characteristic_speeds(
    gsl::not_null<std::array<DataVector, 3>*> char_speeds,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& unit_normal_one_form);

namespace Tags {
template <size_t Dim>
struct CharacteristicSpeedsCompute : Tags::CharacteristicSpeeds<Dim>,
                                     db::ComputeTag {
  using base = Tags::CharacteristicSpeeds<Dim>;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim>>>;

  static void function(
      gsl::not_null<return_type*> char_speeds,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& unit_normal_one_form) {
    characteristic_speeds(char_speeds, unit_normal_one_form);
  }
};
}  // namespace Tags
/// @}

/// @{
/*!
 * \brief Computes characteristic fields from evolved fields for the
 * second-order scalar wave system.
 *
 * The characteristic fields are given in terms of the evolved fields by:
 *
 * \f{align*}
 * v^0_{i} =& (\delta^k_i - n_i n^k) \Phi_{k} \\
 * v^{\pm} =& \Pi \pm n^i \Phi_{i}
 * \f}
 *
 * where \f$n_i\f$ is the unit normal along which the characteristic fields
 * are defined. Note that, unlike the first-order scalar wave system, there is
 * no characteristic field corresponding to \f$\Psi\f$
 * (see \cite Gundlach2005ta).
 *
 * The corresponding characteristic speeds are
 * computed by \ref Tags::CharacteristicSpeedsCompute . The inverse
 * transform, reconstructing \f$(\Pi, \Phi_i)\f$ from the characteristic fields,
 * is computed by
 * \ref Tags::FieldsFromInverseCharacteristicTransformCompute .
 */
template <size_t Dim>
Variables<tmpl::list<Tags::VZero<Dim>, Tags::VPlus, Tags::VMinus>>
characteristic_fields(
    const Scalar<DataVector>& pi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& phi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& unit_normal_one_form);

template <size_t Dim>
void characteristic_fields(
    gsl::not_null<
        Variables<tmpl::list<Tags::VZero<Dim>, Tags::VPlus, Tags::VMinus>>*>
        char_fields,
    const Scalar<DataVector>& pi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& phi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& unit_normal_one_form);

namespace Tags {
template <size_t Dim>
struct CharacteristicFieldsCompute : Tags::CharacteristicFields<Dim>,
                                     db::ComputeTag {
  using base = Tags::CharacteristicFields<Dim>;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<Pi, Phi<Dim>,
                 ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim>>>;

  static void function(
      const gsl::not_null<return_type*> char_fields,
      const Scalar<DataVector>& pi,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& phi,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& unit_normal_one_form) {
    characteristic_fields(char_fields, pi, phi, unit_normal_one_form);
  };
};
}  // namespace Tags
/// @}

/// @{
/*!
 * \brief Reconstruct the fields \f$(\Pi, \Phi_i)\f$ from the characteristic
 * fields.
 *
 * This uses the inverse of the relations in
 * \ref Tags::CharacteristicFieldsCompute :
 *
 * \f{align*}
 * \Pi =& \frac{1}{2}(v^+ + v^-), \\
 * \Phi_{i} =& \frac{1}{2}(v^+ - v^-) n_i + v^0_{i}.
 * \f}
 *
 * The scalar field \f$\Psi\f$ is not reconstructed because it is not part of
 * the characteristic decomposition of the second-order scalar wave system.
 */
template <size_t Dim>
Variables<tmpl::list<Tags::Pi, Tags::Phi<Dim>>>
fields_from_inverse_characteristic_transform(
    const tnsr::i<DataVector, Dim, Frame::Inertial>& v_zero,
    const Scalar<DataVector>& v_plus, const Scalar<DataVector>& v_minus,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& unit_normal_one_form);

template <size_t Dim>
void fields_from_inverse_characteristic_transform(
    gsl::not_null<Variables<tmpl::list<Tags::Pi, Tags::Phi<Dim>>>*>
        evolved_fields,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& v_zero,
    const Scalar<DataVector>& v_plus, const Scalar<DataVector>& v_minus,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& unit_normal_one_form);

namespace Tags {
template <size_t Dim>
struct FieldsFromInverseCharacteristicTransformCompute
    : Tags::FieldsFromInverseCharacteristicTransform<Dim>,
      db::ComputeTag {
  using base = Tags::FieldsFromInverseCharacteristicTransform<Dim>;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<Tags::VZero<Dim>, Tags::VPlus, Tags::VMinus,
                 ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim>>>;

  static void function(
      const gsl::not_null<return_type*> evolved_fields,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& v_zero,
      const Scalar<DataVector>& v_plus, const Scalar<DataVector>& v_minus,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& unit_normal_one_form) {
    fields_from_inverse_characteristic_transform(evolved_fields, v_zero, v_plus,
                                                 v_minus, unit_normal_one_form);
  };
};

struct LargestCharacteristicSpeed : db::SimpleTag {
  using type = double;
};

/// Compute the maximum magnitude of the characteristic speeds.
struct ComputeLargestCharacteristicSpeed : LargestCharacteristicSpeed,
                                           db::ComputeTag {
  using argument_tags = tmpl::list<>;
  using return_type = double;
  using base = LargestCharacteristicSpeed;
  SPECTRE_ALWAYS_INLINE static constexpr void function(
      const gsl::not_null<double*> speed) {
    *speed = 1.0;
  }
};
}  // namespace Tags
/// @}
}  // namespace SecondOrderScalarWave
