// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "AlgorithmSingleton.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "Options/Options.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Printf.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"

namespace Actions {
struct Observe {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static void apply(db::DataBox<DbTags>& /*box*/,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) {
    const auto solution =
        Parallel::get<OptionTags::AnalyticSolution<gr::Solutions::KerrSchild>>(
            cache);

    const auto mass = solution.mass();
    const auto spin = solution.dimensionless_spin();
    const auto center = solution.center();
    Parallel::printf("Kerr Schild Black Hole Quantities\n");
    Parallel::printf("=================================\n\n");
    Parallel::printf("Mass = %1.15f\n", mass);
    Parallel::printf("Spin = [%1.15f, %1.15f, %1.15f]\n", spin[0], spin[1],
                     spin[2]);
    Parallel::printf("Center = [%1.15f, %1.15f, %1.15f]\n\n", center[0],
                     center[1], center[2]);

    // For now, choose a single place and time to evaluate quantities
    // (later, read in a domain and use it to evaluate quantities)
    const double t = 4.4;
    auto x = make_with_value<tnsr::I<double, 3, Frame::Inertial>>(0.0, 0.0);
    get<0>(x) = 4.0;
    get<1>(x) = 3.0;
    get<2>(x) = 2.0;
    Parallel::printf(
        "Quantities at [x, y, z] = [%1.15f, %1.15f, %1.15f] at time "
        "%1.15f:\n\n",
        get<0>(x), get<1>(x), get<2>(x), t);

    const auto vars =
        solution.variables(x, t, gr::Solutions::KerrSchild::tags<double>{});

    // For now, just print the lapse evaluated at the one chosen event
    const auto& lapse = get<gr::Tags::Lapse<double>>(vars);
    Parallel::printf("Lapse = %1.15f\n", get(lapse));
  }
};
}  // namespace Actions

template <class Metavariables>
struct ObserveKerrSchild {
  using analytic_solution_tag =
      OptionTags::AnalyticSolution<gr::Solutions::KerrSchild>;
  using const_global_cache_tag_list = tmpl::list<analytic_solution_tag>;
  using chare_type = Parallel::Algorithms::Singleton;
  using metavariables = Metavariables;
  using action_list = tmpl::list<>;
  using initial_databox = db::DataBox<tmpl::list<>>;
  using options = tmpl::list<>;
  static void initialize(Parallel::CProxy_ConstGlobalCache<
                         Metavariables>& /* global_cache */) noexcept {}
  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) noexcept;
};

template <class Metavariables>
void ObserveKerrSchild<Metavariables>::execute_next_phase(
    const typename Metavariables::Phase /* next_phase */,
    Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) noexcept {
  Parallel::simple_action<Actions::Observe>(
      Parallel::get_parallel_component<ObserveKerrSchild>(
          *(global_cache.ckLocalBranch())));
}

struct Metavars {
  using const_global_cache_tag_list = tmpl::list<>;

  using component_list = tmpl::list<ObserveKerrSchild<Metavars>>;

  static constexpr OptionString help{
      "Output quantities for a Kerr Schild black hole."};

  enum class Phase { Initialization, Execute, Exit };

  static Phase determine_next_phase(const Phase& current_phase,
                                    const Parallel::CProxy_ConstGlobalCache<
                                        Metavars>& /*cache_proxy*/) noexcept {
    return current_phase == Phase::Initialization ? Phase::Execute
                                                  : Phase::Exit;
  }
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
