// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/IndexType.hpp"

/// \cond
template <size_t VolumeDim, typename TargetFrame>
class Domain;
/// \endcond

namespace Initialization {
/// \ingroup InitializationGroup
/// \brief %Tags used during initialization of parallel components.
namespace Tags {
struct InitialTime : db::SimpleTag {
  static std::string name() noexcept { return "InitialTime"; }
  using type = double;
};

struct InitialTimeDelta : db::SimpleTag {
  static std::string name() noexcept { return "InitialTimeDelta"; }
  using type = double;
};

struct InitialSlabSize : db::SimpleTag {
  static std::string name() noexcept { return "InitialSlabSize"; }
  using type = double;
};
}  // namespace Tags
}  // namespace Initialization
