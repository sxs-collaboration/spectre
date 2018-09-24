// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/IndexType.hpp"

/// \cond
class DataVector;

namespace RelativisticEuler {
namespace Valencia {
namespace Tags {
struct TildeD;
struct TildeTau;
template <size_t Dim, typename Fr = Frame::Inertial>
struct TildeS;
}  // namespace Tags
}  // namespace Valencia
}  // namespace RelativisticEuler
/// \endcond
