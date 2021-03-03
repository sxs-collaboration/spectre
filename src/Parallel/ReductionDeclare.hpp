// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Forward-declares the public-facing Reduction classes. This file is needed to
/// avoid circular dependencies in Main.ci, which uses these classes in its
/// declarations.

#pragma once

/// \cond
namespace Parallel {
template <class T, class InvokeCombine, class InvokeFinal,
          class InvokeFinalExtraArgsIndices>
struct ReductionDatum;

template <class... Ts>
struct ReductionData;
}  // namespace Parallel
/// \endcond
