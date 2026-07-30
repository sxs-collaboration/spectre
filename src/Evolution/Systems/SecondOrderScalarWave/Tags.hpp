// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/SecondOrderScalarWave/TagsDeclarations.hpp"
#include "Utilities/TMPL.hpp"

class DataVector;

namespace SecondOrderScalarWave::Tags {

/*!
 * \brief The scalar field.
 */
struct Psi : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/*!
 * \brief The negative time derivative of the scalar field.
 * \details If \f$\Psi\f$ is the scalar field then
 * \f$\Pi = -\partial_t \Psi\f$.
 */
struct Pi : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/*!
 * \brief Auxiliary variable, the spatial derivative of the scalar field.
 * \details If \f$\Psi\f$ is the scalar field then
 * \f$\Phi_i = \partial_i \Psi\f$.
 */
template <size_t Dim>
struct Phi : db::SimpleTag {
  using type = tnsr::i<DataVector, Dim, Frame::Inertial>;
};

/// The contraction \f$n^i\Phi_i\f$ with the face normal.
struct NormalDotPhi : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/// The scalar field multiplied by the face normal, \f$\Psi n_i\f$.
template <size_t Dim>
struct PsiTimesNormal : db::SimpleTag {
  using type = tnsr::i<DataVector, Dim, Frame::Inertial>;
};

/// @{
/// \brief Tags corresponding to the characteristic fields of the second-order
/// scalar-wave system.
template <size_t Dim>
struct VZero : db::SimpleTag {
  using type = tnsr::i<DataVector, Dim, Frame::Inertial>;
};
struct VPlus : db::SimpleTag {
  using type = Scalar<DataVector>;
};
struct VMinus : db::SimpleTag {
  using type = Scalar<DataVector>;
};
/// @}

/// The characteristic speeds corresponding, in order, to `VZero`, `VPlus`, and
/// `VMinus`.
template <size_t Dim>
struct CharacteristicSpeeds : db::SimpleTag {
  using type = std::array<DataVector, 3>;
};

/// The characteristic fields `VZero`, `VPlus`, and `VMinus` packaged together.
template <size_t Dim>
struct CharacteristicFields : db::SimpleTag {
  using type = Variables<tmpl::list<VZero<Dim>, VPlus, VMinus>>;
};

/// The fields \f$(\Pi, \Phi_i)\f$ reconstructed from the
/// characteristic fields.
template <size_t Dim>
struct FieldsFromInverseCharacteristicTransform : db::SimpleTag {
  using type = Variables<tmpl::list<Pi, Phi<Dim>>>;
};
}  // namespace SecondOrderScalarWave::Tags
