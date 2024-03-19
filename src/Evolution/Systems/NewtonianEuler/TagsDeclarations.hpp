// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/IndexType.hpp"

/// \cond
namespace NewtonianEuler {
namespace Tags {

struct MassDensityCons;
template <size_t Dim, typename Fr = Frame::Inertial>
struct MomentumDensity;
struct EnergyDensity;
template <typename DataType>
struct SoundSpeed;

template <size_t Dim>
struct CharacteristicSpeeds;
struct VMinus;
template <size_t Dim>
struct VMomentum;
struct VPlus;

template <typename DataType>
struct InternalEnergyDensity;
template <typename DataType>
struct KineticEnergyDensity;
template <typename DataType>
struct MachNumber;
template <typename DataType, size_t Dim, typename Fr = Frame::Inertial>
struct RamPressure;
template <typename DataType>
struct SpecificKineticEnergy;

template <size_t Dim>
struct SourceTerm;
}  // namespace Tags
}  // namespace NewtonianEuler
/// \endcond
