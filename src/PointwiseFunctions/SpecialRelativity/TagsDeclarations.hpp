// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/IndexType.hpp"

/// \cond
class DataVector;

namespace sr {
namespace Tags {

template <typename DataType>
struct LorentzFactor;
template <typename DataType, size_t Dim, typename Fr>
struct LorentzFactorCompute;
template <typename DataType>
struct LorentzFactorSquared;
template <typename DataType, size_t Dim, typename Fr = Frame::Inertial>
struct SpatialVelocity;
template <typename DataType, size_t Dim, typename Fr = Frame::Inertial>
struct SpatialVelocityOneForm;
template <typename DataType>
struct SpatialVelocitySquared;
}  // namespace Tags
}  // namespace sr
/// \endcond
