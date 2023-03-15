// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Options/String.hpp"

namespace Parallel::OptionTags {
/*!
 * \brief Option group for all things related to parallelization.
 *
 * It is possible this group will need to be used in a library that cannot
 * depend on the `Parallel` library. In that case, this struct should just be
 * forward declared.
 */
struct Parallelization {
  static constexpr Options::String help = {
      "Options related to parallelization aspects of the simulation."};
};
}  // namespace Parallel::OptionTags
