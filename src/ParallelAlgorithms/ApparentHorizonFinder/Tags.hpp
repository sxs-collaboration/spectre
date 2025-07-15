// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <deque>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/LinkedMessageId.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/OptionTags.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Storage.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace control_system::OptionTags {
struct WriteDataToDisk;
}  // namespace control_system::OptionTags
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
/*!
 * \brief Verbosity of horizon finder
 */
struct Verbosity : db::SimpleTag {
  using type = ::Verbosity;
};

/*!
 * \brief Holds a `::FastFlow` object. Needs to be reset after each horizon
 * find.
 */
struct FastFlow : db::SimpleTag {
  using type = ::FastFlow;
};

/*!
 * \brief Tag that holds the current time.
 *
 * \details The value of this tag is `std::nullopt` if the current time isn't
 * set.
 */
struct CurrentTime : db::SimpleTag {
  using type = std::optional<LinkedMessageId<double>>;
};

/*!
 * \brief List of times waiting for previous horizon finds to finish
 * before they can be started.
 */
struct PendingTimes : db::SimpleTag {
  using type = std::set<LinkedMessageId<double>>;
};

/*!
 * \brief Tag that holds all completed times
 */
struct CompletedTimes : db::SimpleTag {
  using type = std::set<LinkedMessageId<double>>;
};

/*!
 * \brief Holds potential dependency for apparent horizon callbacks.
 */
struct Dependency : db::SimpleTag {
  using type = std::optional<std::string>;
};

/*!
 * \brief Storage of all variables (volume or interpolated) for all times of the
 * horizon finder.
 */
template <typename Fr>
struct Storage : db::SimpleTag {
  using type = std::unordered_map<LinkedMessageId<double>,
                                  ah::Storage::SingleTimeStorage<Fr>>;
};

/*!
 * \brief Deque of `ah::Storage::PreviousSurface`s.
 */
template <typename Fr>
struct PreviousSurfaces : db::SimpleTag {
  using type = std::deque<ah::Storage::PreviousSurface<Fr>>;
};

/*!
 * \brief Global cache tag that holds horizon finder options
 */
template <typename HorizonMetavars>
struct ApparentHorizonOptions : db::SimpleTag {
  using type = HorizonOptions<typename HorizonMetavars::frame>;
  using option_tags =
      tmpl::list<OptionTags::ApparentHorizonOptions<HorizonMetavars>>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& option) { return option; }
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

/// Simple tag for whether to write the centers of the horizons to disk.
struct ObserveCenters : db::SimpleTag {
  using type = bool;
  using option_tags = tmpl::list<control_system::OptionTags::WriteDataToDisk>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& option) { return option; }
};
}  // namespace ah::Tags
