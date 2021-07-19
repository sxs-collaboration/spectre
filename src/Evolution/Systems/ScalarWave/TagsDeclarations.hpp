// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

namespace ScalarWave {
struct Psi;
struct Pi;
template <size_t Dim>
struct Phi;

/// \brief Tags for the ScalarWave evolution system
namespace Tags {
struct ConstraintGamma2;

template <size_t Dim>
struct OneIndexConstraint;
template <size_t Dim>
struct TwoIndexConstraint;

struct VPsi;
template <size_t Dim>
struct VZero;
struct VPlus;
struct VMinus;

template <size_t Dim>
struct CharacteristicSpeeds;
template <size_t Dim>
struct CharacteristicFields;
template <size_t Dim>
struct EvolvedFieldsFromCharacteristicFields;
template <size_t Dim>
struct EnergyDensity;
}  // namespace Tags
}  // namespace ScalarWave
