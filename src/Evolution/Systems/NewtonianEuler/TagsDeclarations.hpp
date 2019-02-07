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
}  // namespace Tags
}  // namespace NewtonianEuler
/// \endcond
