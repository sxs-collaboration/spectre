// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines DataBox tags for the curved scalar wave system

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/CurvedScalarWave/TagsDeclarations.hpp"

class DataVector;

namespace CurvedScalarWave {
struct Psi : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "Psi"; }
};

struct Pi : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "Pi"; }
};

template <size_t Dim>
struct Phi : db::SimpleTag {
  using type = tnsr::i<DataVector, Dim>;
  static std::string name() noexcept { return "Phi"; }
};

namespace Tags {
struct ConstraintGamma1 : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "ConstraintGamma1"; }
};

struct ConstraintGamma2 : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "ConstraintGamma2"; }
};

// @{
/// \brief Tags corresponding to the characteristic fields of the
/// scalar-wave system in curved spacetime.
///
/// \details For details on how these are defined and computed, \see
/// CharacteristicSpeedsCompute
struct VPsi : db::SimpleTag {
  using type = Scalar<DataVector>;
};
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
// @}

template <size_t Dim>
struct CharacteristicSpeeds : db::SimpleTag {
  using type = std::array<DataVector, 4>;
};

template <size_t Dim>
struct CharacteristicFields : db::SimpleTag {
  using type = Variables<tmpl::list<VPsi, VZero<Dim>, VPlus, VMinus>>;
};

template <size_t Dim>
struct EvolvedFieldsFromCharacteristicFields : db::SimpleTag {
  using type = Variables<tmpl::list<Psi, Pi, Phi<Dim>>>;
};
}  // namespace Tags
}  // namespace CurvedScalarWave
