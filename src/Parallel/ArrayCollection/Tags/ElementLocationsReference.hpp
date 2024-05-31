// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <unordered_map>

#include "DataStructures/DataBox/Tag.hpp"
#include "Parallel/ArrayCollection/Tags/ElementLocations.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
template <size_t Dim>
class ElementId;
namespace Parallel {
class NodeLock;
}  // namespace Parallel
/// \endcond

namespace Parallel::Tags {
/// \brief The node (location) where different elements are.
///
/// This should be in the DgElementArrayMember's DataBox.
///
/// Implementation note: This should point to the ElementLocations located in
/// the nodegroup's DataBox.
template <size_t Dim, typename Metavariables,
          typename DgElementCollectionComponent>
struct ElementLocationsReference : ElementLocations<Dim>, db::ReferenceTag {
 private:
  struct GetReference {
    using return_type = const typename ElementLocations<Dim>::type&;

    template <typename ParallelComponent, typename DbTagList>
    static return_type apply(
        db::DataBox<DbTagList>& box,
        const gsl::not_null<Parallel::NodeLock*> /*node_lock*/) {
      return db::get_mutable_reference<ElementLocations<Dim>>(
          make_not_null(&box));
    }
  };

 public:
  using base = ElementLocations<Dim>;
  using type = typename base::type;
  using argument_tags =
      tmpl::list<Parallel::Tags::GlobalCacheImpl<Metavariables>>;

  static const type& get(
      const Parallel::GlobalCache<Metavariables>* const& cache) {
    auto& parallel_comp =
        Parallel::get_parallel_component<DgElementCollectionComponent>(*cache);
    return Parallel::local_synchronous_action<GetReference>(parallel_comp);
  }
};
}  // namespace Parallel::Tags
