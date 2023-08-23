// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Utilities/TMPL.hpp"

namespace Parallel::Tags {
/// \cond
template <typename Index>
struct ArrayIndexImpl;
template <typename Metavariables>
struct GlobalCacheProxy;
template <typename Metavariables>
struct MetavariablesImpl;
/// \endcond

/// \brief List of tags for mutable items that are automatically added to
/// the DataBox of a DistributedObject
///
/// \details It is the responsibility of DistributedObject to initialize the
/// mutable items corresponding to these tags.
template <typename Metavariables, typename Index>
using distributed_object_tags =
    tmpl::list<Tags::MetavariablesImpl<Metavariables>,
               Tags::ArrayIndexImpl<Index>,
               Tags::GlobalCacheProxy<Metavariables>>;
}  // namespace Parallel::Tags
