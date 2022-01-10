// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"

/// \brief Tags for the curved scalar wave system
namespace CurvedScalarWave::Tags {
struct Psi;
struct Pi;
template <size_t Dim>
struct Phi;

struct ConstraintGamma1;
struct ConstraintGamma2;

struct VPsi;
template <size_t Dim>
struct VZero;
struct VPlus;
struct VMinus;

template <size_t Dim>
struct CharacteristicSpeeds;
struct LargestCharacteristicSpeed;
template <size_t Dim>
struct CharacteristicFields;
template <size_t Dim>
struct EvolvedFieldsFromCharacteristicFields;
}  // namespace CurvedScalarWave::Tags
