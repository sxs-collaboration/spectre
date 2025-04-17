// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"

/// \cond
class FastFlow;
namespace ylm {
template <typename Frame>
class Strahlkorper;
}  // namespace ylm
/// \endcond

/*!
 * \brief Tags for the apparent horizon finder.
 */
namespace ah::Tags {
struct FastFlow : db::SimpleTag {
  using type = ::FastFlow;
};

/*!
 * \brief Tag that holds the strahlkorper of the previous FastFlow iteration
 * (not the strahlkorper of the entire previous horizon find.)
 */
template <typename Frame>
struct PreviousIterationStrahlkorper : db::SimpleTag {
  using type = ylm::Strahlkorper<Frame>;
};

/*!
 * \brief Tag to hold the number of failed interpolations to a surface during
 * iterations of the FastFlow algorithm.
 */
struct FailedInterpolationIterations : db::SimpleTag {
  using type = size_t;
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
