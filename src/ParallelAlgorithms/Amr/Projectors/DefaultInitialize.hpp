// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <utility>

#include "ParallelAlgorithms/Amr/Protocols/Projector.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t Dim>
class Element;
template <size_t Dim>
class Mesh;
/// \endcond

namespace amr::projectors {

/// \brief Value initialize the items corresponding to Tags
///
/// There is a specialization for `DefaultInitialize<tmpl::list<Tags...>>` that
/// can be used if a `tmpl::list` is available.
///
/// \details For each item corresponding to each tag, value initialize
/// the item by setting it equal to an object constructed with an
/// empty initializer.  This is the default state of mutable items in
/// a DataBox if they are neither set from input file options, nor
/// mutated by initialization actions.
template <typename... Tags>
struct DefaultInitialize : tt::ConformsTo<amr::protocols::Projector> {
  using return_tags = tmpl::list<Tags...>;
  using argument_tags = tmpl::list<>;

  template <size_t Dim>
  static void apply(
      const gsl::not_null<typename Tags::type*>... items,
      const std::pair<Mesh<Dim>, Element<Dim>>& /*old_mesh_and_element*/) {
    expand_pack((*items = std::decay_t<decltype(*items)>{})...);
  }

  template <typename FinalArg>
  static void apply(const gsl::not_null<typename Tags::type*>... /*items*/,
                    const FinalArg& /*parent_or_children_items*/) {
    // Mutable items on newly created elements are already value initialized
  }
};

/// \cond
template <typename... Tags>
struct DefaultInitialize<tmpl::list<Tags...>>
    : public DefaultInitialize<Tags...> {};
/// \endcond
}  // namespace amr::projectors
