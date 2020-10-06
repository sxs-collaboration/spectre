// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "Utilities/PrettyType.hpp"

namespace Convergence {
/// Tags related to the convergence of iterative algorithms
namespace Tags {

/// Identifies a step in an iterative algorithm
template <typename Label>
struct IterationId : db::SimpleTag {
  static std::string name() noexcept {
    return "IterationId(" + pretty_type::short_name<Label>() + ")";
  }
  using type = size_t;
};

}  // namespace Tags
}  // namespace Convergence
