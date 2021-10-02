// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines DataBox tags for scalar wave system

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/ScalarWave/TagsDeclarations.hpp"

class DataVector;

namespace ScalarWave {
struct Psi : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() { return "Psi"; }
};

struct Pi : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() { return "Pi"; }
};

template <size_t Dim>
struct Phi : db::SimpleTag {
  using type = tnsr::i<DataVector, Dim, Frame::Inertial>;
  static std::string name() { return "Phi"; }
};

namespace Tags {
struct ConstraintGamma2 : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() { return "ConstraintGamma2"; }
};

/*!
 * \brief Tag for the one-index constraint of the ScalarWave system
 *
 * For details on how this is defined and computed, see
 * `OneIndexConstraintCompute`.
 */
template <size_t Dim>
struct OneIndexConstraint : db::SimpleTag {
  using type = tnsr::i<DataVector, Dim, Frame::Inertial>;
};
/*!
 * \brief Tag for the two-index constraint of the ScalarWave system
 *
 * For details on how this is defined and computed, see
 * `TwoIndexConstraintCompute`.
 */
template <size_t Dim>
struct TwoIndexConstraint : db::SimpleTag {
  using type = tnsr::ij<DataVector, Dim, Frame::Inertial>;
};

/// @{
/// \brief Tags corresponding to the characteristic fields of the flat-spacetime
/// scalar-wave system.
///
/// \details For details on how these are defined and computed, \see
/// CharacteristicSpeedsCompute
struct VPsi : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() { return "VPsi"; }
};
template <size_t Dim>
struct VZero : db::SimpleTag {
  using type = tnsr::i<DataVector, Dim, Frame::Inertial>;
  static std::string name() { return "VZero"; }
};
struct VPlus : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() { return "VPlus"; }
};
struct VMinus : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() { return "VMinus"; }
};
/// @}

template <size_t Dim>
struct CharacteristicSpeeds : db::SimpleTag {
  using type = std::array<DataVector, 4>;
  static std::string name() { return "CharacteristicSpeeds"; }
};

template <size_t Dim>
struct CharacteristicFields : db::SimpleTag {
  using type = Variables<tmpl::list<VPsi, VZero<Dim>, VPlus, VMinus>>;
  static std::string name() { return "CharacteristicFields"; }
};

template <size_t Dim>
struct EvolvedFieldsFromCharacteristicFields : db::SimpleTag {
  using type = Variables<tmpl::list<Psi, Pi, Phi<Dim>>>;
  static std::string name() { return "EvolvedFieldsFromCharacteristicFields"; }
};

/// The energy density of the scalar wave
template <size_t Dim>
struct EnergyDensity : db::SimpleTag {
  using type = Scalar<DataVector>;
};
}  // namespace Tags
}  // namespace ScalarWave
