// Distributed under the MIT License.
// See LICENSE.txt for details.

// Tests for this file are combined with the ones for
// ObserveFieldsOnHorizon.hpp in
// Test_ObserveFieldsAndTimeSeriesOnHorizon.cpp since they are so similar.

#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/LinkedMessageId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/FastFlow.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Protocols/Callback.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Tags.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace ah::callbacks {
namespace detail {
template <typename T>
struct is_array_of_double : std::false_type {};

template <std::size_t N>
struct is_array_of_double<std::array<double, N>> : std::true_type {};

template <typename... Ts>
auto make_legend(tmpl::list<Ts...> /* meta */) {
  std::vector<std::string> legend = {"Time"};

  [[maybe_unused]] auto append_tags = [&legend](auto tag) {
    using TagType = decltype(tag);
    using ReturnType = typename TagType::type;

    if constexpr (is_array_of_double<ReturnType>::value) {
      constexpr std::array<const char*, 3> suffix = {"_x", "_y", "_z"};
      for (size_t i = 0; i < std::tuple_size<ReturnType>::value; ++i) {
        legend.push_back(db::tag_name<TagType>() + gsl::at(suffix, i));
      }
    } else {
      legend.push_back(db::tag_name<TagType>());
    }
  };

  (append_tags(Ts{}), ...);

  return legend;
}

template <typename DbTags, typename... Ts>
auto make_reduction_data(const db::DataBox<DbTags>& box, double time,
                         tmpl::list<Ts...> /* meta */) {
  return std::make_tuple(time, get<Ts>(box)...);
}

}  // namespace detail

/*!
 * \brief A `ah::protocol::Callback` that outputs a `double` or
 * `std::array<double, N>` quantity on a surface as a time series in the
 * reductions file.
 */
template <typename TagsToObserve, typename HorizonMetavars>
struct ObserveTimeSeriesOnHorizon : tt::ConformsTo<ah::protocols::Callback> {
  template <typename DbTags, typename Metavariables>
  static void apply(const db::DataBox<DbTags>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const FastFlow::Status /*status*/) {
    const auto& time = db::get<ah::Tags::CurrentTime>(box).value();
    auto& proxy = Parallel::get_parallel_component<
        observers::ObserverWriter<Metavariables>>(cache);

    // We call this on proxy[0] because the 0th element of a NodeGroup is
    // always guaranteed to be present.
    Parallel::threaded_action<
        observers::ThreadedActions::WriteReductionDataRow>(
        proxy[0], std::string{"/" + pretty_type::name<HorizonMetavars>()},
        detail::make_legend(TagsToObserve{}),
        detail::make_reduction_data(box, time.id, TagsToObserve{}));
  }
};
}  // namespace ah::callbacks
