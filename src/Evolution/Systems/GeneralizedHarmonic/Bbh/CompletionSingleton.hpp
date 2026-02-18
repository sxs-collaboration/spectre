// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <map>
#include <optional>
#include <set>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Protocols/Mutator.hpp"
#include "DataStructures/LinkedMessageId.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Bbh/CompletionCriteria.hpp"
#include "Parallel/Algorithms/AlgorithmSingletonDeclarations.hpp"
#include "Parallel/ArrayCollection/IsDgElementCollection.hpp"
#include "Parallel/ArrayCollection/SimpleActionOnElement.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "ParallelAlgorithms/Actions/AddSimpleTags.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace gh::bbh {
namespace Tags {
/// Successful AhC finds keyed by temporal id, storing the corresponding `LMax`.
struct CommonHorizonSuccessRecords : db::SimpleTag {
  using type = std::map<LinkedMessageId<double>, size_t>;
  using option_tags = tmpl::list<>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options() { return {}; }
};

/// Reduced constraint maxima keyed by time.
struct ConstraintCheckRecords : db::SimpleTag {
  using key_type = double;
  using mapped_type = std::pair<double, double>;
  using type = std::map<key_type, mapped_type>;
  using option_tags = tmpl::list<>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options() { return {}; }
};

/// Constraint checks already reported at verbose level.
struct ReportedConstraintCheckRecords : db::SimpleTag {
  using type = std::set<gh::bbh::Tags::ConstraintCheckRecords::key_type>;
  using option_tags = tmpl::list<>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options() { return {}; }
};
}  // namespace Tags

namespace Actions {
/// Element simple action that latches completion-request state in the DataBox.
struct SetElementCompletionRequested {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(db::DataBox<DbTags>& box,
                    Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/) {
    db::mutate<gh::bbh::Tags::ElementCompletionRequested>(
        [](const gsl::not_null<bool*> element_completion_requested) {
          *element_completion_requested = true;
        },
        make_not_null(&box));
  }
};
}  // namespace Actions

namespace detail {
template <typename Metavariables>
struct BroadcastCompletionRequestToElements {
  static void apply(Parallel::GlobalCache<Metavariables>& cache) {
    using dg_array = typename Metavariables::gh_dg_element_array;
    if constexpr (Parallel::is_dg_element_collection_v<dg_array>) {
      Parallel::threaded_action<Parallel::Actions::SimpleActionOnElement<
          gh::bbh::Actions::SetElementCompletionRequested, true>>(
          Parallel::get_parallel_component<dg_array>(cache));
    } else {
      Parallel::simple_action<gh::bbh::Actions::SetElementCompletionRequested>(
          Parallel::get_parallel_component<dg_array>(cache));
    }
  }
};

template <typename DbTags, typename Metavariables>
void recompute_completion_state(const gsl::not_null<db::DataBox<DbTags>*> box,
                                Parallel::GlobalCache<Metavariables>& cache) {
  const size_t min_successes =
      Parallel::get<gh::bbh::Tags::MinCommonHorizonSuccessesBeforeChecks>(
          cache);
  const size_t max_successes =
      Parallel::get<gh::bbh::Tags::MaxCommonHorizonSuccesses>(cache);
  const size_t l_max_threshold =
      Parallel::get<gh::bbh::Tags::CommonHorizonLMaxThreshold>(cache);
  const double gauge_constraint_threshold =
      Parallel::get<gh::bbh::Tags::GaugeConstraintLinfThreshold>(cache);
  const double three_index_constraint_threshold =
      Parallel::get<gh::bbh::Tags::ThreeIndexConstraintLinfThreshold>(cache);
  const bool verbose =
      Parallel::get<gh::bbh::Tags::ConstraintCheckVerbose>(cache);
  if (max_successes < min_successes) {
    ERROR_NO_TRACE("MaxCommonHorizonSuccesses ("
                   << max_successes << ") must be >= "
                   << "MinCommonHorizonSuccessesBeforeChecks (" << min_successes
                   << ").");
  }

  const auto& horizon_records =
      db::get<gh::bbh::Tags::CommonHorizonSuccessRecords>(*box);
  const size_t success_count = horizon_records.size();
  bool lmax_latched = false;
  std::optional<std::pair<double, size_t>> first_lmax_match{};
  for (const auto& [time_id, l_max] : horizon_records) {
    if (l_max <= l_max_threshold) {
      lmax_latched = true;
      first_lmax_match = std::pair{time_id.id, l_max};
      break;
    }
  }

  std::optional<double> first_gauge_match_time{};
  std::optional<double> first_three_index_match_time{};
  size_t successes_up_to_constraint_time = 0;
  auto horizon_it = horizon_records.begin();
  const auto& constraint_records =
      db::get<gh::bbh::Tags::ConstraintCheckRecords>(*box);
  const auto& reported_constraint_records =
      db::get<gh::bbh::Tags::ReportedConstraintCheckRecords>(*box);
  std::vector<gh::bbh::Tags::ConstraintCheckRecords::key_type>
      newly_reported_constraint_records{};
  for (const auto& [constraint_time, maxima] : constraint_records) {
    while (horizon_it != horizon_records.end() and
           horizon_it->first.id <= constraint_time) {
      ++successes_up_to_constraint_time;
      ++horizon_it;
    }
    if (successes_up_to_constraint_time < min_successes) {
      continue;
    }
    if (verbose and not reported_constraint_records.contains(constraint_time)) {
      Parallel::printf(
          "BBH completion constraint check at t=%.16f: "
          "Linf(GaugeConstraint)=%.16e (threshold %.16e), "
          "Linf(ThreeIndexConstraint)=%.16e (threshold %.16e).\n",
          constraint_time, maxima.first, gauge_constraint_threshold,
          maxima.second, three_index_constraint_threshold);
      newly_reported_constraint_records.push_back(constraint_time);
    }
    if (not first_gauge_match_time.has_value() and
        maxima.first >= gauge_constraint_threshold) {
      first_gauge_match_time = constraint_time;
    }
    if (not first_three_index_match_time.has_value() and
        maxima.second >= three_index_constraint_threshold) {
      first_three_index_match_time = constraint_time;
    }
  }
  const bool horizon_completion_requested =
      success_count >= min_successes and
      (success_count >= max_successes or lmax_latched);
  const bool completion_requested = horizon_completion_requested or
                                    first_gauge_match_time.has_value() or
                                    first_three_index_match_time.has_value();

  const bool old_gauge_exceeded =
      db::get<gh::bbh::Tags::GaugeConstraintExceeded>(*box);
  const bool old_three_index_exceeded =
      db::get<gh::bbh::Tags::ThreeIndexConstraintExceeded>(*box);
  const bool old_lmax_latched =
      db::get<gh::bbh::Tags::CommonHorizonLMaxBelowOrEqualThreshold>(*box);
  const size_t old_success_count =
      db::get<gh::bbh::Tags::CommonHorizonSuccessCount>(*box);
  const bool old_completion_requested =
      db::get<gh::bbh::Tags::CompletionRequested>(*box);
  db::mutate<gh::bbh::Tags::GaugeConstraintExceeded,
             gh::bbh::Tags::ThreeIndexConstraintExceeded,
             gh::bbh::Tags::CommonHorizonLMaxBelowOrEqualThreshold,
             gh::bbh::Tags::CommonHorizonSuccessCount,
             gh::bbh::Tags::CompletionRequested>(
      [&first_gauge_match_time, &first_three_index_match_time, &lmax_latched,
       &success_count, &completion_requested](
          const gsl::not_null<bool*> gauge_constraint_exceeded_flag,
          const gsl::not_null<bool*> three_index_constraint_exceeded_flag,
          const gsl::not_null<bool*> lmax_latched_flag,
          const gsl::not_null<size_t*> common_horizon_success_count,
          const gsl::not_null<bool*> completion_requested_flag) {
        *gauge_constraint_exceeded_flag = first_gauge_match_time.has_value();
        *three_index_constraint_exceeded_flag =
            first_three_index_match_time.has_value();
        *lmax_latched_flag = lmax_latched;
        *common_horizon_success_count = success_count;
        *completion_requested_flag = completion_requested;
      },
      box);
  if (not newly_reported_constraint_records.empty()) {
    db::mutate<gh::bbh::Tags::ReportedConstraintCheckRecords>(
        [&newly_reported_constraint_records](
            const gsl::not_null<
                gh::bbh::Tags::ReportedConstraintCheckRecords::type*>
                records) {
          for (const auto& key : newly_reported_constraint_records) {
            records->insert(key);
          }
        },
        box);
  }
  if (old_success_count < min_successes and success_count >= min_successes) {
    Parallel::printf(
        "BBH completion criterion armed: AhC successes reached %zu (minimum "
        "required: %zu).\n",
        success_count, min_successes);
  }
  if (not old_lmax_latched and lmax_latched and first_lmax_match.has_value()) {
    Parallel::printf(
        "BBH completion criterion met at t=%.16f: AhC Lmax=%zu <= %zu.\n",
        first_lmax_match->first, first_lmax_match->second, l_max_threshold);
  }
  if (not old_gauge_exceeded and first_gauge_match_time.has_value()) {
    Parallel::printf(
        "BBH completion criterion met at t=%.16f: "
        "Linf(GaugeConstraint) >= %.16e.\n",
        *first_gauge_match_time, gauge_constraint_threshold);
  }
  if (not old_three_index_exceeded and
      first_three_index_match_time.has_value()) {
    Parallel::printf(
        "BBH completion criterion met at t=%.16f: "
        "Linf(ThreeIndexConstraint) >= %.16e.\n",
        *first_three_index_match_time, three_index_constraint_threshold);
  }
  if (not old_completion_requested and completion_requested) {
    Parallel::printf("BBH completion criteria request latched.\n");
    BroadcastCompletionRequestToElements<Metavariables>::apply(cache);
  }
}

struct InitializeCompletionState : tt::ConformsTo<db::protocols::Mutator> {
  using return_tags =
      tmpl::list<gh::bbh::Tags::GaugeConstraintExceeded,
                 gh::bbh::Tags::ThreeIndexConstraintExceeded,
                 gh::bbh::Tags::CommonHorizonLMaxBelowOrEqualThreshold,
                 gh::bbh::Tags::CommonHorizonSuccessCount,
                 gh::bbh::Tags::CompletionRequested,
                 gh::bbh::Tags::CommonHorizonSuccessRecords,
                 gh::bbh::Tags::ConstraintCheckRecords,
                 gh::bbh::Tags::ReportedConstraintCheckRecords>;
  using argument_tags = tmpl::list<>;

  static void apply(
      const gsl::not_null<bool*> gauge_constraint_exceeded,
      const gsl::not_null<bool*> three_index_constraint_exceeded,
      const gsl::not_null<bool*> common_horizon_lmax_below_or_equal_threshold,
      const gsl::not_null<size_t*> common_horizon_success_count,
      const gsl::not_null<bool*> completion_requested,
      const gsl::not_null<gh::bbh::Tags::CommonHorizonSuccessRecords::type*>
          common_horizon_success_records,
      const gsl::not_null<gh::bbh::Tags::ConstraintCheckRecords::type*>
          constraint_check_records,
      const gsl::not_null<gh::bbh::Tags::ReportedConstraintCheckRecords::type*>
          reported_constraint_check_records) {
    *gauge_constraint_exceeded = false;
    *three_index_constraint_exceeded = false;
    *common_horizon_lmax_below_or_equal_threshold = false;
    *common_horizon_success_count = 0;
    *completion_requested = false;
    common_horizon_success_records->clear();
    constraint_check_records->clear();
    reported_constraint_check_records->clear();
  }
};
}  // namespace detail

namespace Actions {
/// Records a successful AhC find and updates BBH completion state.
struct RecordCommonHorizonSuccess {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(db::DataBox<DbTags>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const LinkedMessageId<double>& temporal_id,
                    const size_t l_max) {
    auto& common_horizon_success_records =
        db::get_mutable_reference<gh::bbh::Tags::CommonHorizonSuccessRecords>(
            make_not_null(&box));
    const bool inserted =
        common_horizon_success_records.emplace(temporal_id, l_max).second;
    if (not inserted) {
      ERROR("Duplicate common-horizon completion record for temporal id "
            << temporal_id << ".");
    }
    detail::recompute_completion_state(make_not_null(&box), cache);
  }
};

/// Reduction target callback that records reduced constraint maxima and updates
/// BBH completion state.
struct ProcessConstraintMaxima {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(db::DataBox<DbTags>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/, const double time,
                    const double max_gauge_linf,
                    const double max_three_index_linf) {
    auto& constraint_check_records =
        db::get_mutable_reference<gh::bbh::Tags::ConstraintCheckRecords>(
            make_not_null(&box));
    const bool inserted =
        constraint_check_records
            .emplace(time,
                     gh::bbh::Tags::ConstraintCheckRecords::mapped_type{
                         max_gauge_linf, max_three_index_linf})
            .second;
    if (not inserted) {
      ERROR("Duplicate BBH completion constraint-max record at t=" << time
                                                                   << ".");
    }
    detail::recompute_completion_state(make_not_null(&box), cache);
  }
};

/// Initializes the element-local completion-request tag.
struct InitializeElementCompletionRequested
    : tt::ConformsTo<db::protocols::Mutator> {
  using return_tags = tmpl::list<gh::bbh::Tags::ElementCompletionRequested>;
  using argument_tags = tmpl::list<>;

  static void apply(const gsl::not_null<bool*> element_completion_requested) {
    *element_completion_requested = false;
  }
};
}  // namespace Actions

template <class Metavariables>
struct CompletionSingleton {
  using chare_type = Parallel::Algorithms::Singleton;
  static constexpr bool checkpoint_data = true;
  using metavariables = Metavariables;
  using const_global_cache_tags =
      tmpl::list<gh::bbh::Tags::MinCommonHorizonSuccessesBeforeChecks,
                 gh::bbh::Tags::MaxCommonHorizonSuccesses,
                 gh::bbh::Tags::GaugeConstraintLinfThreshold,
                 gh::bbh::Tags::ThreeIndexConstraintLinfThreshold,
                 gh::bbh::Tags::CommonHorizonLMaxThreshold,
                 gh::bbh::Tags::ConstraintCheckVerbose>;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<Initialization::Actions::AddSimpleTags<
                     gh::bbh::detail::InitializeCompletionState>,
                 Parallel::Actions::TerminatePhase>>>;
  using simple_tags_from_options = tmpl::list<>;

  static void execute_next_phase(
      const Parallel::Phase next_phase,
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *Parallel::local_branch(global_cache);
    Parallel::get_parallel_component<CompletionSingleton<Metavariables>>(
        local_cache)
        .start_phase(next_phase);
  }
};
}  // namespace gh::bbh
