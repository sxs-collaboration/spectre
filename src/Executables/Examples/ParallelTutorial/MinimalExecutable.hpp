// Distributed under the MIT License.
// See LICENSE.txt for details.
/// \cond
#pragma once

#include <vector>

#include "Options/Options.hpp"
#include "Utilities/TMPL.hpp"

namespace Parallel {
template <typename Metavariables>
class CProxy_GlobalCache;
}  // namespace Parallel

/// [metavariables_definition]
struct Metavariables {
  using component_list = tmpl::list<>;

  enum class Phase { Initialization, Exit };

  static Phase determine_next_phase(
      const Phase& /*current_phase*/,
      const Parallel::CProxy_GlobalCache<
          Metavariables>& /*cache_proxy*/) noexcept {
    return Phase::Exit;
  }

  static constexpr OptionString help{"A minimal executable"};
};
/// [metavariables_definition]

/// [executable_example_charm_init]
static const std::vector<void (*)()> charm_init_node_funcs{};
static const std::vector<void (*)()> charm_init_proc_funcs{};
/// [executable_example_charm_init]
/// \endcond
