// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/IndexType.hpp"

class DataVector;

namespace Ccz4 {

/// \brief Tags for the CCZ4 formulation of Einstein equations
namespace Tags {
template <typename DataType = DataVector>
struct ConformalFactor;
}  // namespace Tags

/// \brief Input option tags for the generalized harmonic evolution system
namespace OptionTags {
struct Group;
}  // namespace OptionTags
}  // namespace Ccz4
