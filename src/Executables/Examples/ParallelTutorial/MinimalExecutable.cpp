// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <array>

#include "Options/String.hpp"
#include "Parallel/CharmMain.tpp"
#include "Parallel/Phase.hpp"
#include "Utilities/TMPL.hpp"

// [metavariables_definition]
struct Metavariables {
  using component_list = tmpl::list<>;

  static constexpr std::array<Parallel::Phase, 2> default_phase_order{
      {Parallel::Phase::Initialization, Parallel::Phase::Exit}};

  static constexpr Options::String help{"A minimal executable"};
};
// [metavariables_definition]

// [main_function]
extern "C" void CkRegisterMainModule() {
  Parallel::charmxx::register_main_module<Metavariables>();
}
// [main_function]
