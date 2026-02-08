// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/ObservationBox.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/Protocols/ReductionDataFormatter.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "IO/Observer/TypeOfObservation.hpp"
#include "Options/String.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Utilities/TMPL.hpp"

namespace PUP {
class er;
}  // namespace PUP

namespace amr::Events {
namespace detail {
using AmrStatsReductionData = Parallel::ReductionData<
    // Observation value / time
    Parallel::ReductionDatum<double, funcl::AssertEqual<>>,
    // Total number of elements
    Parallel::ReductionDatum<size_t, funcl::Plus<>>,
    // Total number of grid points
    Parallel::ReductionDatum<size_t, funcl::Plus<>>,
    // Number of grid points per dimension: total, min, max
    Parallel::ReductionDatum<std::vector<size_t>,
                             funcl::ElementWise<funcl::Plus<>>>,
    Parallel::ReductionDatum<std::vector<size_t>,
                             funcl::ElementWise<funcl::Min<>>>,
    Parallel::ReductionDatum<std::vector<size_t>,
                             funcl::ElementWise<funcl::Max<>>>>;

struct FormatAmrStatsOutput
    : tt::ConformsTo<observers::protocols::ReductionDataFormatter> {
  using reduction_data = AmrStatsReductionData;
  std::string operator()(double time, size_t total_num_elements,
                         size_t total_num_points,
                         const std::vector<size_t>& num_points_per_dim,
                         const std::vector<size_t>& min_points_per_dim,
                         const std::vector<size_t>& max_points_per_dim) const;
  // NOLINTNEXTLINE
  void pup(PUP::er& p);
};
}  // namespace detail

/*!
 * \ingroup AmrGroup
 * \brief Observes AMR statistics, such as number of elements and grid points
 */
template <size_t Dim>
class ObserveAmrStats
    : public SPECTRE_CHARM_DERIVED(SINGLE_ARG(ObserveAmrStats<Dim>), Event) {
 public:
  struct PrintToTerminal {
    using type = bool;
    static constexpr Options::String help = {
        "Whether to print reduction info to terminal."};
  };
  struct ObservePerCore {
    using type = bool;
    static constexpr Options::String help = {
        "Also write reduction observations per-core in a file per-node."};
  };

  /// \cond
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ObserveAmrStats);  // NOLINT
  /// \endcond

  using options = tmpl::list<PrintToTerminal, ObservePerCore>;
  static constexpr Options::String help = {"Observe AMR statistics"};

  ObserveAmrStats();
  ObserveAmrStats(bool print_to_terminal, bool observe_per_core);

  using observed_reduction_data_tags = observers::make_reduction_data_tags<
      tmpl::list<detail::AmrStatsReductionData>>;

  using compute_tags_for_observation_box = tmpl::list<>;
  using argument_tags = tmpl::list<domain::Tags::Mesh<Dim>>;
  using return_tags = tmpl::list<>;

  template <typename Metavariables, typename ParallelComponent>
  void operator()(const Mesh<Dim>& mesh,
                  Parallel::GlobalCache<Metavariables>& cache,
                  const ElementId<Dim>& element_id,
                  const ParallelComponent* const /*meta*/,
                  const ObservationValue& observation_value) const {
    const auto& mesh_extents = mesh.extents();
    const std::vector<size_t> mesh_extents_vec{mesh_extents.begin(),
                                               mesh_extents.end()};
    detail::AmrStatsReductionData reduction_data{observation_value.value,
                                                 1_st,
                                                 mesh.number_of_grid_points(),
                                                 mesh_extents_vec,
                                                 mesh_extents_vec,
                                                 mesh_extents_vec};
    std::vector<std::string> legend{observation_value.name, "NumElements",
                                    "TotalNumPoints"};
    std::vector<std::string> legend_num_p;
    std::vector<std::string> legend_min_p;
    std::vector<std::string> legend_max_p;
    legend_num_p.reserve(Dim);
    legend_min_p.reserve(Dim);
    legend_max_p.reserve(Dim);
    for (size_t d = 0; d < Dim; ++d) {
      legend_num_p.push_back("NumPointsPerDim_" + std::to_string(d));
      legend_min_p.push_back("MinPointsPerDim_" + std::to_string(d));
      legend_max_p.push_back("MaxPointsPerDim_" + std::to_string(d));
    }
    legend.insert(legend.end(), legend_num_p.begin(), legend_num_p.end());
    legend.insert(legend.end(), legend_min_p.begin(), legend_min_p.end());
    legend.insert(legend.end(), legend_max_p.begin(), legend_max_p.end());

    auto& local_observer = *Parallel::local_branch(
        Parallel::get_parallel_component<
            tmpl::conditional_t<Parallel::is_nodegroup_v<ParallelComponent>,
                                observers::ObserverWriter<Metavariables>,
                                observers::Observer<Metavariables>>>(cache));
    observers::ObservationId observation_id{observation_value.value,
                                            subfile_path_ + ".dat"};
    Parallel::ArrayComponentId array_component_id{
        std::add_pointer_t<ParallelComponent>{nullptr},
        Parallel::ArrayIndex<ElementId<Dim>>(element_id)};
    auto formatter = print_to_terminal_
                         ? std::make_optional(detail::FormatAmrStatsOutput{})
                         : std::nullopt;
    if constexpr (Parallel::is_nodegroup_v<ParallelComponent>) {
      const std::optional<int> observe_with_core_id =
          observe_per_core_
              ? std::make_optional(Parallel::my_node<int>(local_observer))
              : std::nullopt;
      Parallel::threaded_action<
          observers::ThreadedActions::CollectReductionDataOnNode>(
          local_observer, std::move(observation_id),
          std::move(array_component_id), subfile_path_, std::move(legend),
          std::move(reduction_data), std::move(formatter),
          observe_with_core_id);
    } else {
      Parallel::simple_action<observers::Actions::ContributeReductionData>(
          local_observer, std::move(observation_id),
          std::move(array_component_id), subfile_path_, std::move(legend),
          std::move(reduction_data), std::move(formatter), observe_per_core_);
    }
  }

  using observation_registration_tags = tmpl::list<>;

  std::optional<
      std::pair<observers::TypeOfObservation, observers::ObservationKey>>
  get_observation_type_and_key_for_registration() const {
    return {{observers::TypeOfObservation::Reduction,
             observers::ObservationKey(subfile_path_ + ".dat")}};
  }

  using is_ready_argument_tags = tmpl::list<>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  bool is_ready(Parallel::GlobalCache<Metavariables>& /*cache*/,
                const ArrayIndex& /*array_index*/,
                const Component* const /*meta*/) const {
    return true;
  }

  bool needs_evolved_variables() const override { return false; }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

 private:
  bool print_to_terminal_{false};
  bool observe_per_core_{false};
  std::string subfile_path_{"Amr"};
};
}  // namespace amr::Events
