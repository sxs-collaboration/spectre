// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines tags related to Time quantities

#pragma once

#include <deque>
#include <memory>
#include <tuple>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Options/Options.hpp"
#include "Time/History.hpp"
#include "Time/Time.hpp"
#include "Time/TimeId.hpp"
#include "Utilities/TMPL.hpp"

class TimeStepper;

namespace Tags {

/// \ingroup DataBoxTagsGroup
/// \ingroup TimeGroup
/// \brief Tag for ::TimeId for the algorithm state
struct TimeId : db::DataBoxTag {
  static constexpr db::DataBoxString label = "TimeId";
  using type = ::TimeId;
};

/// \ingroup DataBoxTagsGroup
/// \ingroup TimeGroup
/// \brief Tag for step size
struct TimeStep : db::DataBoxTag {
  static constexpr db::DataBoxString label = "TimeStep";
  using type = ::TimeDelta;
};

namespace TimeTags_detail {
inline ::Time time_from_id(const ::TimeId& id) noexcept { return id.time; }
}  // namespace TimeTags_detail

/// \ingroup DataBoxTagsGroup
/// \ingroup TimeGroup
/// \brief Tag for compute item for current ::Time (from TimeId)
struct Time : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "Time";
  static constexpr auto function = TimeTags_detail::time_from_id;
  using argument_tags = tmpl::list<TimeId>;
};

/// \ingroup DataBoxTagsGroup
/// \ingroup TimeGroup
/// \brief Prefix for TimeStepper history
///
/// \tparam Tag tag for the variables
/// \tparam DtTag tag for the time derivative of the variables
template <typename Tag, typename DtTag>
struct HistoryEvolvedVariables : db::DataBoxPrefix {
  static constexpr db::DataBoxString label = "HistoryEvolvedVariables";
  using tag = Tag;
  using type = TimeSteppers::History<db::item_type<Tag>, db::item_type<DtTag>>;
};

/// \ingroup DataBoxTagsGroup
/// \ingroup TimeGroup
/// Prefix for TimeStepper boundary history
///
/// \tparam Key type identifying a boundary
/// \tparam Tag tag for boundary variables
template <typename Key, typename Tag, typename Hash = std::hash<Key>>
struct HistoryBoundaryVariables : db::DataBoxPrefix {
  static constexpr db::DataBoxString label = "HistoryBoundaryVariables";
  using tag = Tag;
  using type = std::unordered_map<Key, db::item_type<Tag>, Hash>;
};

}  // namespace Tags

namespace CacheTags {

/// \ingroup CacheTagsGroup
/// \ingroup TimeGroup
/// \brief The final time
struct FinalTime {
  using type = double;
  static constexpr OptionString help{"The final time"};
};

/// \ingroup CacheTagsGroup
/// \ingroup TimeGroup
/// \brief The ::TimeStepper
struct TimeStepper {
  using type = std::unique_ptr<::TimeStepper>;
  static constexpr OptionString help{"The time stepper"};
};

}  // namespace CacheTags

namespace OptionTags {

/// \ingroup OptionTagsGroup
/// \ingroup TimeGroup
/// \brief The time at which to start the simulation
struct InitialTime {
  using type = double;
  static constexpr OptionString help = {
      "The time at which the evolution is started."};
  static type default_value() { return 0.0; }
};

/// \ingroup OptionTagsGroup
/// \ingroup TimeGroup
/// \brief The initial time step taken by the time stepper. This may be
/// overridden by an adaptive stepper
struct DeltaT {
  using type = double;
  static constexpr OptionString help = {"The initial time step size."};
};
}  // namespace OptionTags
