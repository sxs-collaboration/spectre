// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Parallel/ArrayIndex.hpp"

#include "Parallel/Algorithms/AlgorithmGroup.decl.h"

namespace Parallel {
namespace Algorithms {

/*!
 * \ingroup ParallelGroup
 * \brief A struct that stores the charm++ types relevant for a particular group
 * component
 *
 * \details The type traits are:
 * - `cproxy`: the charm++ proxy.
 *   See https://charm.readthedocs.io/en/latest/faq/manual.html#what-is-a-proxy
 * - `cbase`: the charm++ base class. See
 *   https://charm.readthedocs.io/en/latest/charm++/manual.html#chare-objects
 * - `algorithm_type`: the chare type (`AlgorithmGroup`) for the group
 *   component.
 * - `ckindex`: A charm++ chare index object. Useful for obtaining entry
 *   method indices that are needed for creating callbacks. See
 *   https://charm.readthedocs.io/en/latest/charm++/manual.html#creating-a-ckcallback-object
 * - `cproxy_section`: The charm++ section proxy class. See
 *   https://charm.readthedocs.io/en/latest/charm++/manual.html?#sections-subsets-of-a-chare-array-group
 */
struct Group {
  template <typename ParallelComponent,
            typename SpectreArrayIndex>
  using cproxy = CProxy_AlgorithmGroup<ParallelComponent,
                                       SpectreArrayIndex>;

  template <typename ParallelComponent,
            typename SpectreArrayIndex>
  using cbase = CBase_AlgorithmGroup<ParallelComponent,
                                     SpectreArrayIndex>;

  template <typename ParallelComponent,
            typename SpectreArrayIndex>
  using algorithm_type = AlgorithmGroup<ParallelComponent,
                                        SpectreArrayIndex>;

  template <typename ParallelComponent,
            typename SpectreArrayIndex>
  using ckindex = CkIndex_AlgorithmGroup<ParallelComponent,
                                         SpectreArrayIndex>;

  template <typename ParallelComponent, typename SpectreArrayIndex>
  using cproxy_section =
      CProxySection_AlgorithmGroup<ParallelComponent, SpectreArrayIndex>;
};
}  // namespace Algorithms
}  // namespace Parallel
