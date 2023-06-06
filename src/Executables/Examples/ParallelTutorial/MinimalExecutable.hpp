// Distributed under the MIT License.
// See LICENSE.txt for details.
/// \cond
#pragma once

#include <vector>

#include "Options/String.hpp"
#include "Parallel/Phase.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Parallel {
template <typename Metavariables>
class CProxy_GlobalCache;
}  // namespace Parallel

/// [metavariables_definition]
struct Metavariables {
  using component_list = tmpl::list<>;

  static constexpr std::array<Parallel::Phase, 2> default_phase_order{
      {Parallel::Phase::Initialization, Parallel::Phase::Exit}};

  static constexpr Options::String help{"A minimal executable"};
};
/// [metavariables_definition]

/// [executable_example_charm_init]
static const std::vector<void (*)()> charm_init_node_funcs{};
static const std::vector<void (*)()> charm_init_proc_funcs{};
/// [executable_example_charm_init]
/// \endcond
