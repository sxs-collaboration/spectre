// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/IndexType.hpp"

/// \cond
namespace NewtonianEuler {
namespace Tags {
template <size_t Dim>
struct CharacteristicSpeeds;
template <typename DataType>
struct MassDensity;
template <typename DataType>
struct MassDensityCons;
template <typename DataType, size_t Dim, typename Fr = Frame::Inertial>
struct MomentumDensity;
template <typename DataType>
struct EnergyDensity;
template <typename DataType, size_t Dim, typename Fr = Frame::Inertial>
struct Velocity;
template <typename DataType>
struct SpecificInternalEnergy;
template <typename DataType>
struct Pressure;
template <typename DataType>
struct SoundSpeed;
template <typename DataType>
struct SoundSpeedCompute;
template <typename DataType>
struct SoundSpeedSquared;
template <typename DataType>
struct SoundSpeedSquaredCompute;
}  // namespace Tags
}  // namespace NewtonianEuler
/// \endcond
