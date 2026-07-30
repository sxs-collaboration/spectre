// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

/// \brief Tags for the second-order ScalarWave evolution system
namespace SecondOrderScalarWave::Tags {
struct Psi;
struct Pi;
template <size_t Dim>
struct Phi;
struct NormalDotPhi;
template <size_t Dim>
struct PsiTimesNormal;

template <size_t Dim>
struct VZero;
struct VPlus;
struct VMinus;

template <size_t Dim>
struct CharacteristicSpeeds;
template <size_t Dim>
struct CharacteristicFields;
template <size_t Dim>
struct FieldsFromInverseCharacteristicTransform;
}  // namespace SecondOrderScalarWave::Tags
