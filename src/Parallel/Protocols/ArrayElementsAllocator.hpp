// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "Utilities/TMPL.hpp"

namespace Parallel::protocols {
/*!
 * \brief Conforming types implement a strategy to create elements for array
 * parallel components
 *
 * Conforming classes must provide the following type aliases:
 *
 * - `array_allocation_tags<ParallelComponent>`: A `tmpl::list` of tags that are
 *   needed to perform the allocation. These tags will be parsed from input-file
 *   options (see \ref dev_guide_parallelization_parallel_components). The array
 *   parallel component will be passed in as a template parameter.
 *
 * Conforming classes must implement the following static member functions:
 *
 * - `apply<ParallelComponent>`: This function is responsible for creating the
 *   array elements. It has the same signature as the `allocate_array` function
 *   (see \ref dev_guide_parallelization_parallel_components), but takes the
 *   array parallel component as an additional first template parameter.
 *
 * See `elliptic::DefaultElementsAllocator` for an example implementation of
 * this protocol.
 */
struct ArrayElementsAllocator {
  template <typename ConformingType>
  struct test {};
};
}  // namespace Parallel::protocols
