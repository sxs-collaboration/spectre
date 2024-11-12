// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"

/// \cond
class FastFlow;
/// \endcond

/*!
 * \brief Tags for the apparent horizon finder.
 */
namespace ah::Tags {
struct FastFlow : db::SimpleTag {
  using type = ::FastFlow;
};

/// Base tag for whether or not to write the centers of the horizons to disk.
/// Most likely to be used in the `ObserveCenters` post horizon find callback
///
/// Other things can control whether the horizon centers are output by defining
/// their own simple tag from this base tag.
struct ObserveCentersBase : db::BaseTag {};

/// Simple tag for whether to write the centers of the horizons to disk.
/// Currently this tag is not creatable by options
struct ObserveCenters : ObserveCentersBase, db::SimpleTag {
  using type = bool;
};
}  // namespace ah::Tags
