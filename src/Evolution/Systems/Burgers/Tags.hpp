// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"

namespace Burgers {
/// \brief Tags for the Burgers system
namespace Tags {
struct U : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/// The characteristic speeds
struct CharacteristicSpeeds : db::SimpleTag {
  using type = std::array<DataVector, 1>;
};
}  // namespace Tags
}  // namespace Burgers
