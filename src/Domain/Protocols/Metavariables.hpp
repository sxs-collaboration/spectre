// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

namespace domain::protocols {
/*!
 * \brief Compile-time options for a Domain.
 *
 * A class conforming to this protocol is placed in the metavariables to choose
 * compile-time options for creating domains. The conforming class must provide
 * the following static member variables:
 * - `bool enable_time_dependent_maps`: Whether or not domains may have
 * time-dependent maps. When set to `true` domain creators may require
 * additional input-file options related to the time dependence. When set to
 * `false` the domains are static, e.g. for elliptic problems.
 *
 * Here is an example of a class that conforms to this protocol:
 *
 * \snippet Domain/Test_Protocols.cpp domain_metavariables_example
 *
 */
struct Metavariables {
  template <typename ConformingType>
  struct test {
    using enable_time_dependent_maps_type = const bool;
    using enable_time_dependent_maps_return_type =
        decltype(ConformingType::enable_time_dependent_maps);
    static_assert(std::is_same_v<enable_time_dependent_maps_type,
                                 enable_time_dependent_maps_return_type>,
                  "The metavariable 'enable_time_dependent_maps' should be a "
                  "static constexpr bool.");
  };
};
}  // namespace domain::protocols
