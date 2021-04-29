// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/TagsDeclarations.hpp"
#include "Evolution/Tags.hpp"

/// \cond
class DataVector;
/// \endcond

namespace grmhd {
namespace GhValenciaDivClean {
/// %Tags for the Valencia formulation of the ideal GRMHD equations
/// with divergence cleaning.
namespace Tags {

/// The characteristic speeds
struct CharacteristicSpeeds : db::SimpleTag {
  using type = std::array<DataVector, 13>;
};

/// The largest characteristic speed at any point
struct LargestCharacteristicSpeed : db::SimpleTag {
  using type = double;
};
}  // namespace Tags
}  // namespace GhValenciaDivClean
}  // namespace grmhd
