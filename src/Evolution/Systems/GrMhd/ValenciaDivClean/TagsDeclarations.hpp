// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/IndexType.hpp"

/// \cond
class DataVector;

namespace grmhd {
namespace ValenciaDivClean {
namespace Tags {
struct CharacteristicSpeeds;
struct ConstraintDampingParameter;
struct TildeD;
struct TildeTau;
template <typename Fr = Frame::Inertial>
struct TildeS;
template <typename Fr = Frame::Inertial>
struct TildeB;
struct TildePhi;
}  // namespace Tags
}  // namespace ValenciaDivClean
}  // namespace grmhd
/// \endcond
