// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <array>

#include "DataStructures/DataBox/Tag.hpp"
#include "Helpers/Parallel/RoundRobinArrayElements.hpp"
#include "Options/String.hpp"
#include "Parallel/Algorithms/AlgorithmArray.hpp"
#include "Parallel/CharmMain.tpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Printf.hpp"
#include "Utilities/TMPL.hpp"

namespace {
namespace OptionTags {
struct Lines {
  using type = int;
  static constexpr Options::String help{"Number of lines printed"};
};
struct NumberOfElements {
  using type = int;
  static constexpr Options::String help{"Number of elements"};
};
}  // namespace OptionTags

namespace Tags {
struct Lines : db::SimpleTag {
  using type = int;
  using option_tags = tmpl::list<OptionTags::Lines>;

  static constexpr bool pass_metavariables = false;
  static int create_from_options(const int lines) { return lines; }
};
struct NumberOfElements : db::SimpleTag {
  using type = int;
  using option_tags = tmpl::list<OptionTags::NumberOfElements>;

  static constexpr bool pass_metavariables = false;
  static int create_from_options(const int number_of_elements) {
    return number_of_elements;
  }
};
}  // namespace Tags

namespace Actions {
struct PrintMessage {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(db::DataBox<DbTags>& box,
                    const Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/) {
    std::stringstream ss{};
    const auto id = db::get<Parallel::Tags::ArrayIndex>(box);
    const auto lines = Parallel::get<Tags::Lines>(cache);
    const auto my_proc = sys::my_proc();
    const auto my_node = sys::my_node();
    ss << "Begin atomic print by " << id << " on process " << my_proc
       << " on node " << my_node << "\n";
    for (int i = 0; i < lines; ++i) {
      ss << "Print by " << id << " on process " << my_proc << " on node "
         << my_node << "\n";
    }
    ss << "End atomic print by " << id << " on process " << my_proc
       << " on node " << my_node << "\n";
    Parallel::printf("\n%s\n", ss.str());
  }
};
}  // namespace Actions

template <class Metavariables>
struct TestArray {
  using chare_type = Parallel::Algorithms::Array;
  using metavariables = Metavariables;
  using array_index = int;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization, tmpl::list<>>>;
  using simple_tags_from_options = Parallel::get_simple_tags_from_options<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;
  using array_allocation_tags = tmpl::list<Tags::NumberOfElements>;

  static void allocate_array(
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache,
      const tuples::tagged_tuple_from_typelist<simple_tags_from_options>&
      /*initialization_items*/,
      const tuples::tagged_tuple_from_typelist<array_allocation_tags>&
          array_allocation_items = {},
      const std::unordered_set<size_t>& procs_to_ignore = {}) {
    auto& local_cache = *Parallel::local_branch(global_cache);
    auto& array_proxy =
        Parallel::get_parallel_component<TestArray>(local_cache);
    const auto number_of_elements =
        tuples::get<Tags::NumberOfElements>(array_allocation_items);
    TestHelpers::Parallel::assign_array_elements_round_robin_style(
        array_proxy, static_cast<size_t>(number_of_elements),
        static_cast<size_t>(sys::number_of_procs()), {}, global_cache,
        procs_to_ignore);
  }

  static void execute_next_phase(
      const Parallel::Phase next_phase,
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *Parallel::local_branch(global_cache);
    auto& my_proxy = Parallel::get_parallel_component<TestArray>(local_cache);
    my_proxy.start_phase(next_phase);
    if (next_phase == Parallel::Phase::Execute) {
      Parallel::simple_action<Actions::PrintMessage>(my_proxy);
    }
  }
};

struct TestMetavariables {
  using component_list = tmpl::list<TestArray<TestMetavariables>>;
  using const_global_cache_tags = tmpl::list<Tags::Lines>;

  static constexpr Options::String help =
      "An executable for testing atomic printing in parallel.";

  static constexpr auto default_phase_order =
      std::array{Parallel::Phase::Initialization, Parallel::Phase::Execute,
                 Parallel::Phase::Exit};

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}
};
}  // namespace

extern "C" void CkRegisterMainModule() {
  Parallel::charmxx::register_main_module<TestMetavariables>();
  Parallel::charmxx::register_init_node_and_proc({}, {});
}
