// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/IndexType.hpp"

/// \cond
namespace grmhd {
namespace ValenciaDivClean {
namespace Tags {
struct TildeD;
struct TildeTau;
template <typename Fr = Frame::Inertial>
struct TildeS;
template <typename Fr = Frame::Inertial>
struct TildeB;
struct TildePhi;

struct ConstraintDampingParameter;

struct CharacteristicSpeeds;
struct VDivCleanMinus;
struct VMinus;
struct VMomentum;
struct VPlus;
struct VDivCleanPlus;
}  // namespace Tags
}  // namespace ValenciaDivClean
}  // namespace grmhd
/// \endcond
