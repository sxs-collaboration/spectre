// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "ParallelAlgorithms/Amr/Protocols/Projector.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace amr::protocols {
/// \brief Compile-time information for AMR projectors
///
/// A class conforming to this protocol is placed in the metavariables to
/// provide the following:
/// - `element_array`: The array component on which AMR is performed.
/// - `projectors`: A type list of AMR projectors (each of which must conform to
///   amr::protocols::Projector) that will be applied by:
///     - amr::Actions::InitializeChild and amr::Actions::InitializeParent in
///       order to initialize data on newly created elements.
///     - amr::Actions::AdjustDomain in order to update data on existing
///       elements in case their Mesh or neighbors have changed.
///   In these projectors you must handle _all_ mutable tags in the DataBox,
///   except for a few tags that are handled by the AMR actions themselves
///   (e.g. `domain::Tags::Element`, `domain::Tags::Mesh`, and
///   `domain::Tags::NeighborMesh` are handled by AMR). See
///   `amr::protocols::Projector` for details.
/// - `keep_coarse_grids`: A boolean indicating that AMR should create a
///   completely new grid at each AMR step with an incremented grid index, and
///   keep the old grid around. This is useful for multigrid solvers.
///   If this is true, then the `element_array` must include
///   `::amr::Actions::RegisterElement` in the registration phase action list
///   and in `Metavariables::registration::element_registrars`. You must also
///   ensure to visit `Phase::UpdateSections` in the default phase order after
///   registration, and in each AMR step after
///   `Phase::EvaluateRefinementCriteria` and `Phase::AdjustDomain`.
///   When this is enabled, you can use `amr::Tags::ParentId` and
///   `amr::Tags::ChildIds` to traverse the grid hierarchy. However, you cannot
///   rely on these tags to be up-to-date in the AMR projectors, as they are
///   sometime updated after the projectors are run.
///
/// Here is an example for a class conforming to this protocol:
///
/// \snippet Amr/Test_Protocols.cpp amr_projectors
struct AmrMetavariables {
  template <typename ConformingType>
  struct test {
    using element_array = typename ConformingType::element_array;
    using projectors = typename ConformingType::projectors;
    static_assert(
        tmpl::all<projectors,
                  tt::assert_conforms_to<tmpl::_1, Projector>>::value);
    static constexpr bool keep_coarse_grids = ConformingType::keep_coarse_grids;
  };
};
}  // namespace amr::protocols
