// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/IndexType.hpp"

/// \cond
namespace NewtonianEuler {
namespace Tags {

template <typename DataType>
struct MassDensity;
struct MassDensityCons;
template <size_t Dim, typename Fr = Frame::Inertial>
struct MomentumDensity;
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
struct SoundSpeedSquared;

template <size_t Dim>
struct CharacteristicSpeeds;
struct VMinus;
template <size_t Dim>
struct VMomentum;
struct VPlus;

struct SourceTermBase;
template <typename InitialDataType>
struct SourceTerm;

}  // namespace Tags
}  // namespace NewtonianEuler
/// \endcond
