// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

namespace ah::protocols {
/*!
 * \brief A protocol for a callback to a horizon find.
 *
 * \details Can be used in either `horizon_find_callbacks` or
 * `horizon_find_failure_callbacks` in the `ah::protocols::HorizonMetavars`.
 *
 * A struct conforming to the this protocol must have
 *
 * - An apply function with the signature in the example
 *
 * A struct conforming to this protocol can also optionally specify
 *
 * - a type alias `const_global_cache_tags` that holds global cache tags.
 *
 * \snippet Helpers/ParallelAlgorithms/ApparentHorizonFinder/TestHelpers.hpp HorizonFindCallback
 */
struct Callback {
  template <typename ConformingType>
  struct test {};
};
}  // namespace ah::protocols
