// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines actions ApplyLimiter and SendDataForLimiter

#pragma once

#include <boost/functional/hash.hpp>  // IWYU pragma: keep
#include <cstddef>
#include <map>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Tags.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace SlopeLimiters {
namespace Tags {
/// \ingroup DiscontinuousGalerkinGroup
/// \ingroup SlopeLimitersGroup
/// \brief The inbox tag for limiter communication.
template <typename Metavariables>
struct LimiterCommunicationTag {
  static constexpr size_t volume_dim = Metavariables::system::volume_dim;
  using packaged_data_t = typename Metavariables::limiter::type::PackagedData;
  using temporal_id = db::item_type<typename Metavariables::temporal_id>;
  using type =
      std::map<temporal_id,
               std::unordered_map<
                   std::pair<Direction<volume_dim>, ElementId<volume_dim>>,
                   packaged_data_t,
                   boost::hash<std::pair<Direction<volume_dim>,
                                         ElementId<volume_dim>>>>>;
};
}  // namespace Tags

namespace Actions {
/// \ingroup ActionsGroup
/// \ingroup DiscontinuousGalerkinGroup
/// \ingroup SlopeLimitersGroup
/// \brief Receive limiter data from neighbors, then apply limiter.
///
/// Currently, is not tested for support of:
/// - h-refinement
/// Currently, does not support:
/// - Local time-stepping
///
/// Uses:
/// - ConstGlobalCache:
///   - Metavariables::limiter
/// - DataBox:
///   - Metavariables::limiter::type::limit_argument_tags
///   - Metavariables::temporal_id
///   - Tags::Element<volume_dim>
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies:
///   - Metavariables::limiter::type::limit_tags
///
/// \see SendDataForLimiter
template <typename Metavariables>
struct Limit {
  using const_global_cache_tags = tmpl::list<typename Metavariables::limiter>;

  static_assert(
      not Metavariables::local_time_stepping,
      "Limiter communication actions do not yet support local time stepping");

 public:
  using limiter_comm_tag =
      SlopeLimiters::Tags::LimiterCommunicationTag<Metavariables>;
  using inbox_tags = tmpl::list<limiter_comm_tag>;

  template <typename DbTags, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::ConstGlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    using mutate_tags = typename Metavariables::limiter::type::limit_tags;
    using argument_tags =
        typename Metavariables::limiter::type::limit_argument_tags;

    const auto& limiter = get<typename Metavariables::limiter>(cache);
    const auto& local_temporal_id =
        db::get<typename Metavariables::temporal_id>(box);
    auto& inbox = tuples::get<limiter_comm_tag>(inboxes);
    db::mutate_apply<mutate_tags, argument_tags>(limiter, make_not_null(&box),
                                                 inbox[local_temporal_id]);

    inbox.erase(local_temporal_id);

    return std::forward_as_tuple(std::move(box));
  }

  template <typename DbTags, typename... InboxTags, typename ArrayIndex>
  static bool is_ready(
      const db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/) noexcept {
    constexpr size_t volume_dim = Metavariables::system::volume_dim;
    const auto& element = db::get<::Tags::Element<volume_dim>>(box);
    const auto num_expected = element.neighbors().size();
    // Edge case where we do not receive any data
    if (UNLIKELY(num_expected == 0)) {
      return true;
    }
    const auto& local_temporal_id =
        db::get<typename Metavariables::temporal_id>(box);
    const auto& inbox = tuples::get<limiter_comm_tag>(inboxes);
    const auto& received = inbox.find(local_temporal_id);
    // Check we have at least some data from correct time
    if (received == inbox.end()) {
      return false;
    }
    // Check data was received from each neighbor
    const size_t num_neighbors_received = received->second.size();
    return (num_neighbors_received == num_expected);
  }
};

/// \ingroup ActionsGroup
/// \ingroup DiscontinuousGalerkinGroup
/// \ingroup SlopeLimitersGroup
/// \brief Send local data needed for limiting.
///
/// Currently, is not tested for support of:
/// - h-refinement
/// Currently, does not support:
/// - Local time-stepping
///
/// Uses:
/// - ConstGlobalCache:
///   - Metavariables::limiter
/// - DataBox:
///   - Tags::Element<volume_dim>
///   - Metavariables::limiter::type::package_argument_tags
///   - Metavariables::temporal_id
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies: nothing
///
/// \see ApplyLimiter
template <typename Metavariables>
struct SendData {
  using const_global_cache_tags = tmpl::list<typename Metavariables::limiter>;
  using limiter_comm_tag =
      SlopeLimiters::Tags::LimiterCommunicationTag<Metavariables>;

  static_assert(
      not Metavariables::local_time_stepping,
      "Limiter communication actions do not yet support local time stepping");

  template <typename DbTags, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::ConstGlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    constexpr size_t volume_dim = Metavariables::system::volume_dim;

    auto& receiver_proxy =
        Parallel::get_parallel_component<ParallelComponent>(cache);
    const auto& element = db::get<::Tags::Element<volume_dim>>(box);
    const auto& temporal_id = db::get<typename Metavariables::temporal_id>(box);
    const auto& limiter = get<typename Metavariables::limiter>(cache);

    for (const auto& direction_neighbors : element.neighbors()) {
      const auto& direction = direction_neighbors.first;
      const size_t dimension = direction.dimension();
      const auto& neighbors_in_direction = direction_neighbors.second;
      ASSERT(neighbors_in_direction.size() == 1,
             "h-adaptivity is not supported yet.\nDirection: "
                 << direction << "\nDimension: " << dimension
                 << "\nNeighbors:\n"
                 << neighbors_in_direction);
      const auto& orientation = neighbors_in_direction.orientation();
      const auto direction_from_neighbor = orientation(direction.opposite());

      using argument_tags =
          typename Metavariables::limiter::type::package_argument_tags;
      const auto packaged_data = db::apply<argument_tags>(
          [&limiter](const auto&... args) noexcept {
            // Note: orientation is received as last element of pack `args`
            typename Metavariables::limiter::type::PackagedData pack{};
            limiter.package_data(make_not_null(&pack), args...);
            return pack;
          },
          box, orientation);

      for (const auto& neighbor : neighbors_in_direction) {
        Parallel::receive_data<limiter_comm_tag>(
            receiver_proxy[neighbor], temporal_id,
            std::make_pair(
                std::make_pair(direction_from_neighbor, element.id()),
                packaged_data));

      }  // loop over neighbors_in_direction
    }    // loop over element.neighbors()

    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace Actions
}  // namespace SlopeLimiters
