// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace db {
template <typename DbTagList>
class DataBox;
}  // namespace db
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
/// \endcond

namespace Parallel {
namespace detail {
template <typename Metavariables, typename ParallelComponent,
          typename = std::void_t<>>
struct has_registration_list : std::false_type {};

template <typename Metavariables, typename ParallelComponent>
struct has_registration_list<
    Metavariables, ParallelComponent,
    std::void_t<typename Metavariables::template registration_list<
        ParallelComponent>::type>> : std::true_type {};

template <typename Metavariables, typename ParallelComponent>
constexpr bool has_registration_list_v =
    has_registration_list<Metavariables, ParallelComponent>::value;

template <typename Metavariables, typename ParallelComponent>
using registration_list =
    typename Metavariables::template registration_list<ParallelComponent>::type;
}  // namespace detail

/// @{
/// \brief (De)register an array element for specified actions
///
/// \details Array elements are (de)registered with actions on components that
/// need to know which elements are contributing data before the action can be
/// executed.  If array elements are migrated (e.g. during load balancing),
/// or are created/destroyed (e.g. during adaptive mesh refinement), these
/// functions must be called in order to (un)register (old) new elements.
/// The list of registration actions is obtained from
/// `Metavariables::registration_list::type`.
template <typename ParallelComponent, typename DbTagList,
          typename Metavariables, typename ArrayIndex>
void deregister_element(db::DataBox<DbTagList>& box,
                        Parallel::GlobalCache<Metavariables>& cache,
                        const ArrayIndex& array_index) {
  if constexpr (detail::has_registration_list_v<Metavariables,
                                                ParallelComponent>) {
    using registration_list =
        typename Metavariables::template registration_list<
            ParallelComponent>::type;
    tmpl::for_each<registration_list>(
        [&box, &cache, &array_index](auto registration_v) {
          using registration = typename decltype(registration_v)::type;
          registration::template perform_deregistration<ParallelComponent>(
              box, cache, array_index);
        });
  }
}

template <typename ParallelComponent, typename DbTagList,
          typename Metavariables, typename ArrayIndex>
void register_element(db::DataBox<DbTagList>& box,
                      Parallel::GlobalCache<Metavariables>& cache,
                      const ArrayIndex& array_index) {
  if constexpr (detail::has_registration_list_v<Metavariables,
                                                ParallelComponent>) {
    using registration_list =
        typename Metavariables::template registration_list<
            ParallelComponent>::type;
    tmpl::for_each<registration_list>(
        [&box, &cache, &array_index](auto registration_v) {
          using registration = typename decltype(registration_v)::type;
          registration::template perform_registration<ParallelComponent>(
              box, cache, array_index);
        });
  }
}
/// @}
}  // namespace Parallel
