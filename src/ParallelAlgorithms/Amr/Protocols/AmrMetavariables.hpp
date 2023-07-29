// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "ParallelAlgorithms/Amr/Protocols/Projector.hpp"

namespace amr::protocols {
/// \brief Compile-time information for AMR projectors
///
/// A class conforming to this protocol is placed in the metavariables to
/// provide the list of AMR projectors (each of which must conform to
/// amr::protocols::Projector) that will be applied by:
/// - amr::Actions::InitializeChild and amr::Actions::InitializeParent in order
///   to initialize data on newly created elements.
/// - amr::Actions::AdjustDomain in order to update data on existing elements in
///   case their Mesh or neighbors have changed.
///
///
/// Here is an example for a class conforming to this protocol:
///
/// \snippet Amr/Test_Protocols.cpp amr_projectors
struct AmrMetavariables {
  template <typename ConformingType>
  struct test {
    using projectors = typename ConformingType::projectors;
    static_assert(
        tmpl::all<projectors,
                  tt::assert_conforms_to<tmpl::_1, Projector>>::value);
  };
};
}  // namespace amr::protocols
