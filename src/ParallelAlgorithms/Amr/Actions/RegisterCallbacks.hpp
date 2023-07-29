// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <deque>
#include <vector>

#include "Parallel/Callback.hpp"
#include "ParallelAlgorithms/Amr/Actions/CollectDataFromChildren.hpp"
#include "ParallelAlgorithms/Amr/Actions/Component.hpp"
#include "ParallelAlgorithms/Amr/Actions/CreateChild.hpp"
#include "ParallelAlgorithms/Amr/Actions/SendDataToChildren.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace amr {
template <typename Metavariables, typename Component>
void register_callbacks() {
  using ArrayIndex = typename Component::array_index;
  register_classes_with_charm(
      tmpl::list<
          Parallel::SimpleActionCallback<
              amr::Actions::CreateChild,
              CProxy_AlgorithmSingleton<amr::Component<Metavariables>, int>,
              CProxy_AlgorithmArray<Component, ArrayIndex>, ArrayIndex,
              std::vector<ArrayIndex>, size_t>,
          Parallel::SimpleActionCallback<
              amr::Actions::SendDataToChildren,
              CProxyElement_AlgorithmArray<Component, ArrayIndex>,
              std::vector<ArrayIndex>>,
          Parallel::SimpleActionCallback<
              amr::Actions::CollectDataFromChildren,
              CProxyElement_AlgorithmArray<Component, ArrayIndex>, ArrayIndex,
              std::deque<ArrayIndex>>>{});
}
}  // namespace amr
