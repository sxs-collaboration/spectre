// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Options/Options.hpp"

namespace CacheTags {
/*!
 * \ingroup CacheTagsGroup
 * \brief The global cache tag that retrieves the parameters for the numerical
 * flux from the input file
 */
template <typename NumericalFluxType>
struct NumericalFluxParams {
  static constexpr OptionString help = "The options for the numerical flux";
  using type = NumericalFluxType;
};
}  // namespace CacheTags
