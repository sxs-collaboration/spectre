// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "Parallel/Protocols/ElementRegistrar.hpp"
#include "Utilities/TMPL.hpp"

namespace Parallel::protocols {
/*!
 * \brief Conforming types provide compile-time information for registering and
 * deregistering array elements with other parallel components
 *
 * A class conforming to this protocol is placed in the metavariables to provide
 * a list of element registrars for each array parallel component. The element
 * registrars in the list must conform to the
 * `Parallel::protocols::ElementRegistrar` protocol.
 *
 * The class conforming to this protocol must provide the following type alias:
 *
 * - `element_registrars`: A `tmpl::map` from parallel components to their list
 *   of element registrars. The element registrars in the typelist must conform
 *   to the `Parallel::protocols::ElementRegistrar` protocol.
 *
 * Here is an example implementation of this protocol:
 *
 * \snippet Test_InitializeParent.cpp registration_metavariables
 *
 * \note We may consider retrieving the list of element registrars directly from
 * the `Parallel::Phase::Registration` phase of the parallel component instead
 * of requiring the metavariables to provide a separate list.
 */
struct RegistrationMetavariables {
  template <typename ConformingType>
  struct test {
    using element_registrars = typename ConformingType::element_registrars;
  };
};
}  // namespace Parallel::protocols
