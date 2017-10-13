// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines tags related to Time quantities

#pragma once

#include <deque>
#include <memory>
#include <tuple>

#include "DataStructures/DataBoxTag.hpp"
#include "Options/Options.hpp"
#include "Time/Time.hpp"
#include "Time/TimeId.hpp"
#include "Utilities/TMPL.hpp"

class TimeStepper;

namespace Tags {

/// \ingroup DataBoxTags
/// \ingroup TimeGroup
/// \brief Tag for ::TimeId for the algorithm state
struct TimeId : db::DataBoxTag {
  static constexpr db::DataBoxString_t label = "TimeId";
  using type = ::TimeId;
};

/// \ingroup DataBoxTags
/// \ingroup TimeGroup
/// \brief Tag for step size
struct TimeStep : db::DataBoxTag {
  static constexpr db::DataBoxString_t label = "TimeStep";
  using type = ::TimeDelta;
};

namespace TimeTags_detail {
inline ::Time time_from_id(const ::TimeId& id) noexcept { return id.time; }
}  // namespace TimeTags_detail

/// \ingroup DataBoxTags
/// \ingroup TimeGroup
/// \brief Tag for compute item for current ::Time (from TimeId)
struct Time : db::ComputeItemTag {
  static constexpr db::DataBoxString_t label = "Time";
  static constexpr auto function = TimeTags_detail::time_from_id;
  using argument_tags = tmpl::list<TimeId>;
};

/// \ingroup DataBoxTags
/// \ingroup TimeGroup
/// \brief Prefix for TimeStepper history
///
/// \tparam Tag tag for the variables
/// \tparam DtTag tag for the time derivative of the variables
template <typename Tag, typename DtTag>
struct HistoryEvolvedVariables : db::DataBoxPrefix {
  static constexpr db::DataBoxString_t label = "HistoryEvolvedVariables";
  using tag = Tag;
  using type =
      std::deque<std::tuple<::Time, db::item_type<Tag>, db::item_type<DtTag>>>;
};

}  // namespace Tags

namespace CacheTags {

/// \ingroup CacheTags
/// \ingroup TimeGroup
/// \brief The final time
struct FinalTime {
  using type = double;
  static constexpr OptionString_t help{"The final time"};
};

/// \ingroup CacheTags
/// \ingroup TimeGroup
/// \brief The ::TimeStepper
struct TimeStepper {
  using type = std::unique_ptr<::TimeStepper>;
  static constexpr OptionString_t help{"The time stepper"};
};

}  // namespace CacheTags
