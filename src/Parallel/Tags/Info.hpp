// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>

#include "DataStructures/DataBox/Tag.hpp"

/// \cond
namespace sys {
class Info;
}  // namespace sys
/// \endcond

namespace Parallel::Tags {
/// \ingroup ParallelGroup
/// Tag to retrieve a concrete ::sys::Info.
struct Info : db::SimpleTag {
  using type = std::unique_ptr<::sys::Info>;
};
}  // namespace Parallel::Tags
