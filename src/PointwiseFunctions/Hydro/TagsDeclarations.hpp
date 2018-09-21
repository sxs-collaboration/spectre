// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/IndexType.hpp"

/// \cond
class DataVector;

namespace hydro {
namespace Tags {

template <typename DataType>
struct AlfvenSpeedSquared;
template <typename DataType, size_t Dim, typename Fr = Frame::Inertial>
struct ComovingMagneticField;
template <typename DataType>
struct DivergenceCleaningField;
template <bool IsRelativistic, size_t ThermodynamicDim>
struct EquationOfState;
template <typename DataType>
struct LorentzFactor;
template <typename DataType, size_t Dim, typename Fr = Frame::Inertial>
struct MagneticField;
template <typename DataType>
struct MagneticPressure;
template <typename DataType>
struct Pressure;
template <typename DataType>
struct RestMassDensity;
template <typename DataType>
struct SoundSpeedSquared;
template <typename DataType, size_t Dim, typename Fr = Frame::Inertial>
struct SpatialVelocity;
template <typename DataType, size_t Dim, typename Fr = Frame::Inertial>
struct SpatialVelocityOneForm;
template <typename DataType>
struct SpatialVelocitySquared;
template <typename DataType>
struct SpecificEnthalpy;
template <typename DataType>
struct SpecificInternalEnergy;
}  // namespace Tags
}  // namespace hydro
/// \endcond
