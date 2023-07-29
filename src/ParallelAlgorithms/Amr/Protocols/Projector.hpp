// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "Utilities/ProtocolHelpers.hpp"

namespace amr::protocols {

/// \brief A DataBox mutator used in AMR actions
///
/// A class conforming to this protocol can be used as a projector in the list
/// of `projectors` for a class conforming to amr::protocols::AmrMetavariables.
/// The conforming class will be used when adaptive mesh refinement occurs to
/// either initialize items on a newly created element of a DgElementArray, or
/// update items on an existing element.
///
/// The conforming class must provide the following:
/// - `return_tags`: A type list of tags corresponding to mutable items in the
///   DataBox that may be modified during refinement.
/// - `argument_tags`: A type list of tags corresponding to items in the DataBox
///   that are not changed, but used to initialize/update the items
///   corresponding to the `return_tags`.
/// - `apply`:  static functions whose return value are void, and that take as
///   arguments:
///      - A `const gsl::not_null<Tag::type*>` for each `Tag` in `return_tags`
///      - A `const db::const_item_type<Tag, BoxTags>` for each `Tag` in
///        `argument_tags`
///      - and one additional argument which is either:
///           - `const std::pair<Mesh<Dim>, Element<Dim>>&` (used by
///             amr::Actions::AdjustDomain)
///           - `const tuples::TaggedTuple<Tags...>&` (used by
///             amr::Actions::InitializeChild)
///           - `const std::unordered_map<ElementId<Dim>,
///                                       tuples::TaggedTuple<Tags...>>&`
///             (used by amr::Actions::InitializeParent)
///
///   The Mesh and Element passed to amr::Actions::AdjustDomain are their
///   values before the grid changes.  The tuples passed to
///   amr::Actions::InitializeChild and amr::Actions::InitializeParent hold the
///   items corresponding to the `DataBox<BoxTags>::mutable_item_creation_tags`
///   of the parent (children) of the child (parent) being initialized.
///
/// \note In amr::Actions::AdjustDomain the projectors are called on
/// all elements that were not h-refined (i.e. split or joined) even
/// if their Mesh did not change.  This allows a projector to mutate a
/// mutable item that depends upon information about the neighboring
/// elements.  Therefore a particular projector may want to check
/// whether or not the Mesh changed before projecting any data.
///
/// For examples, see Initialization::ProjectTimeStepping and
/// evolution::dg::Initialization::ProjectDomain
struct Projector {
  template <typename ConformingType>
  struct test {
    using argument_tags = typename ConformingType::argument_tags;
    using return_tags = typename ConformingType::return_tags;
  };
};
}  // namespace amr::protocols
