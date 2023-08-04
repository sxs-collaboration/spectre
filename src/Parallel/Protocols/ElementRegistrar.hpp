// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

namespace Parallel::protocols {
/*!
 * \brief Conforming types register and deregister array elements with other
 * parallel components
 *
 * For example, array elements may have to register and deregister with
 * interpolators or observers when the elements get created and destroyed during
 * AMR or migrated during load balancing. They may do so by sending messages to
 * the parallel components that notify of the creation and destruction of the
 * array elements.
 *
 * Conforming classes have the following static member functions:
 *
 * ```cpp
 * static void perform_registration<ParallelComponent>(const
 *   db::DataBox<DbTagList>& box, Parallel::GlobalCache<Metavariables>& cache,
 *   const ArrayIndex& array_index)
 *
 * static void perform_deregistration<ParallelComponent>(const
 *   db::DataBox<DbTagList>& box, Parallel::GlobalCache<Metavariables>& cache,
 *   const ArrayIndex& array_index)
 * ```
 *
 * Here is an example implementation of this protocol:
 *
 * \snippet RegistrationHelpers.hpp element_registrar_example
 */
struct ElementRegistrar {
  template <typename ConformingType>
  struct test {};
};
}  // namespace Parallel::protocols
