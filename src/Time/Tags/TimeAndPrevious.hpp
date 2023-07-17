// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/LinkedMessageId.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Tags {
struct Time;
}  // namespace Tags
namespace evolution::Tags {
struct PreviousTriggerTime;
}  // namespace evolution::Tags
/// \endcond

namespace Tags {

/// @{
/// \ingroup TimeGroup
/// \brief Tag for the current and previous time as doubles
///
/// \warning The previous time is calculated via the value of the
/// ::evolution::Tags::PreviousTriggerTime. Therefore, this tag can only be
/// used in the context of dense triggers as that is where the
/// ::evolution::Tags::PreviousTriggerTime tag is set. Any Events that request
/// this tag in their `argument_tags` type alias, must be triggered by a
/// DenseTrigger.
///
/// \note The Index is just so we can have multiple of this tag in the same
/// DataBox.
template <size_t Index>
struct TimeAndPrevious : db::SimpleTag {
  using type = LinkedMessageId<double>;
  static std::string name() { return "TimeAndPrevious" + get_output(Index); }
};

template <size_t Index>
struct TimeAndPreviousCompute : TimeAndPrevious<Index>, db::ComputeTag {
  using argument_tags =
      tmpl::list<::Tags::Time, ::evolution::Tags::PreviousTriggerTime>;
  using base = TimeAndPrevious<Index>;
  using return_type = LinkedMessageId<double>;

  static void function(
      gsl::not_null<LinkedMessageId<double>*> time_and_previous,
      const double time, const std::optional<double>& previous_time) {
    time_and_previous->id = time;
    time_and_previous->previous = previous_time;
  }
};
/// @}
}  // namespace Tags
