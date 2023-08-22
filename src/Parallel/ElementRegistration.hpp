// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "Parallel/Protocols/ElementRegistrar.hpp"
#include "Parallel/Protocols/RegistrationMetavariables.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/CreateHasTypeAlias.hpp"

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
CREATE_HAS_TYPE_ALIAS(registration)
CREATE_HAS_TYPE_ALIAS_V(registration)
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
/// `Metavariables::registration::element_registrars`.
///
/// \see Parallel::protocols::RegistrationMetavariables
/// \see Parallel::protocols::ElementRegistrar
template <typename ParallelComponent, typename DbTagList,
          typename Metavariables, typename ArrayIndex>
void deregister_element(db::DataBox<DbTagList>& box,
                        Parallel::GlobalCache<Metavariables>& cache,
                        const ArrayIndex& array_index) {
  if constexpr (detail::has_registration_v<Metavariables>) {
    static_assert(tt::assert_conforms_to_v<
                  typename Metavariables::registration,
                  Parallel::protocols::RegistrationMetavariables>);
    using element_registrars =
        typename Metavariables::registration::element_registrars;
    if constexpr (tmpl::has_key<element_registrars, ParallelComponent>::value) {
      tmpl::for_each<tmpl::at<element_registrars, ParallelComponent>>(
          [&box, &cache, &array_index](auto registration_v) {
            using registration = typename decltype(registration_v)::type;
            static_assert(tt::assert_conforms_to_v<
                          registration, Parallel::protocols::ElementRegistrar>);
            registration::template perform_deregistration<ParallelComponent>(
                box, cache, array_index);
          });
    }
  }
}

template <typename ParallelComponent, typename DbTagList,
          typename Metavariables, typename ArrayIndex>
void register_element(db::DataBox<DbTagList>& box,
                      Parallel::GlobalCache<Metavariables>& cache,
                      const ArrayIndex& array_index) {
  if constexpr (detail::has_registration_v<Metavariables>) {
    static_assert(tt::assert_conforms_to_v<
                  typename Metavariables::registration,
                  Parallel::protocols::RegistrationMetavariables>);
    using element_registrars =
        typename Metavariables::registration::element_registrars;
    if constexpr (tmpl::has_key<element_registrars, ParallelComponent>::value) {
      tmpl::for_each<tmpl::at<element_registrars, ParallelComponent>>(
          [&box, &cache, &array_index](auto registration_v) {
            using registration = typename decltype(registration_v)::type;
            static_assert(tt::assert_conforms_to_v<
                          registration, Parallel::protocols::ElementRegistrar>);
            registration::template perform_registration<ParallelComponent>(
                box, cache, array_index);
          });
    }
  }
}
/// @}
}  // namespace Parallel
