// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <pup.h>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "IO/Observer/TypeOfObservation.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/Auto.hpp"
#include "Options/String.hpp"
#include "Parallel/ArrayComponentId.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/Reduction.hpp"
#include "Parallel/TypeTraits.hpp"
#include "ParallelAlgorithms/Events/ReductionScope.hpp"
#include "ParallelAlgorithms/Events/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"
#include "Utilities/TMPL.hpp"

namespace Events {
/// \cond
template <size_t Dim, typename NonTensorComputeTags>
class ObserveGridStats;
/// \endcond

namespace ObserveGridStats_detail {
using ReductionData = Parallel::ReductionData<
    // Observation value
    Parallel::ReductionDatum<double, funcl::AssertEqual<>>,
    // Number of elements
    Parallel::ReductionDatum<size_t, funcl::Plus<>>,
    // Number of grid points
    Parallel::ReductionDatum<size_t, funcl::Plus<>>,
    // Fraction of elements doing FD
    Parallel::ReductionDatum<double, funcl::Plus<>, funcl::Divides<>,
                             std::index_sequence<1>>,
    // Fraction of grid points doing FD
    Parallel::ReductionDatum<double, funcl::Plus<>, funcl::Divides<>,
                             std::index_sequence<2>>,
    // Grid points per dimension: average, min, max
    Parallel::ReductionDatum<
        std::vector<double>, funcl::ElementWise<funcl::Plus<>>,
        funcl::ElementWise<funcl::Divides<>>, std::index_sequence<1>>,
    Parallel::ReductionDatum<std::vector<size_t>,
                             funcl::ElementWise<funcl::Min<>>>,
    Parallel::ReductionDatum<std::vector<size_t>,
                             funcl::ElementWise<funcl::Max<>>>,
    // Refinement levels per dimension: average, min, max
    Parallel::ReductionDatum<
        std::vector<double>, funcl::ElementWise<funcl::Plus<>>,
        funcl::ElementWise<funcl::Divides<>>, std::index_sequence<1>>,
    Parallel::ReductionDatum<std::vector<size_t>,
                             funcl::ElementWise<funcl::Min<>>>,
    Parallel::ReductionDatum<std::vector<size_t>,
                             funcl::ElementWise<funcl::Max<>>>>;

// The column names of the reduction data
std::vector<std::string> legend(const std::string& observation_value_name,
                                size_t dim);

// The contribution of a single element to the reduction
template <size_t Dim>
ReductionData reduction_data(double observation_value,
                             const Mesh<Dim>& observer_mesh,
                             const ElementId<Dim>& element_id);

// The path of the subfile (without extension) that the element contributes
// to, or `std::nullopt` if the element is not observed.
template <size_t Dim>
std::optional<std::string> subfile_path(
    ReductionScope reduction_scope,
    const std::optional<std::vector<std::string>>& blocks_to_observe,
    const Domain<Dim>& domain, const ElementId<Dim>& element_id);
}  // namespace ObserveGridStats_detail

/*!
 * \brief Observe statistics of the computational grid.
 *
 * Each observation writes a row with the following columns:
 * - The observation value (e.g. `Time`)
 * - `NumElements`: The number of elements
 * - `NumPoints`: The number of grid points
 * - `FractionOfElementsFd`: The fraction of elements doing FD
 * - `FractionOfPointsFd`: The fraction of grid points in elements doing FD
 * - `{Avg,Min,Max}PointsPerDim_{d}`: The average, min, and max number of grid
 *   points in each logical dimension `d` over the elements
 * - `{Avg,Min,Max}RefinementLevel_{d}`: The average, min, and max refinement
 *   level in each logical dimension `d` over the elements
 *
 * All averages are over elements, i.e. each element is weighted equally.
 *
 * The grid points are counted on the `Events::Tags::ObserverMesh`, which must
 * be provided by the `NonTensorComputeTags`. For hybrid DG-FD methods this is
 * the subcell mesh in elements doing FD, so elements doing FD contribute their
 * subcell grid points. An element is considered to be doing FD if its
 * `ObserverMesh` has a `Spectral::Basis::FiniteDifference` basis in any
 * dimension. Without subcell the FD fractions are always zero.
 *
 * \par Observing a subset of blocks
 * The `BlocksToObserve` option restricts the observation to the listed blocks
 * and block groups. Set it to `All` to observe the whole domain.
 *
 * \par Reduction scope
 * The `ReductionScope` option (see `Events::ReductionScope`) controls whether
 * the observed elements are reduced together or split up by block:
 * - `WholeDomain`: a single reduction over all observed elements, written to
 *   the subfile `GridStats.dat`. In addition, the same statistics restricted
 *   to the elements on each core are written to the subfile
 *   `Core{core_id}/GridStats.dat` in one H5 file per node (see
 *   `observers::Actions::ContributeReductionData`). This is useful to assess
 *   load-balancing. For nodegroup components the statistics are per node
 *   instead of per core.
 * - `ByBlock`: a separate reduction for each observed block, written to
 *   `GridStats/<BlockName>.dat`.
 * - `ByBlockOrGroup`: a separate reduction for each entry of `BlocksToObserve`,
 *   written to `GridStats/<BlockOrGroupName>.dat`.
 *
 * To observe both the whole domain and individual blocks, add two
 * `ObserveGridStats` events, one with `ReductionScope: WholeDomain` and one
 * with `ReductionScope: ByBlock`.
 */
template <size_t Dim, typename... NonTensorComputeTags>
class ObserveGridStats<Dim, tmpl::list<NonTensorComputeTags...>>
    : public Event {
  static_assert(
      (std::is_base_of_v<::Events::Tags::ObserverMesh<Dim>,
                         NonTensorComputeTags> or
       ...),
      "The NonTensorComputeTags of ObserveGridStats must contain a tag that "
      "derives from Events::Tags::ObserverMesh<Dim>, e.g. "
      "Events::Tags::ObserverMeshCompute<Dim>.");

 public:
  /// \brief A list of block or group names on which to observe.
  ///
  /// Set to `All` to observe everywhere.
  struct BlocksToObserve {
    using type =
        Options::Auto<std::vector<std::string>, Options::AutoLabel::All>;
    static constexpr Options::String help = {
        "A list of block and group names on which to observe."};
  };
  /// \brief Whether to reduce over all observed elements or separately for
  /// each block or group.
  struct ReductionScope {
    using type = ::Events::ReductionScope;
    static constexpr Options::String help = {
        "How to group the observed elements into reductions: WholeDomain, "
        "ByBlock, or ByBlockOrGroup. WholeDomain also writes the statistics "
        "on each core."};
  };

  /// \cond
  explicit ObserveGridStats(CkMigrateMessage* msg) : Event(msg) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ObserveGridStats);  // NOLINT
  /// \endcond

  using options = tmpl::list<BlocksToObserve, ReductionScope>;
  static constexpr Options::String help =
      "Observe statistics of the computational grid, such as the number of "
      "elements and grid points, their resolution and refinement levels, and "
      "the fraction of the grid that is doing finite-difference.";

  ObserveGridStats() = default;
  ObserveGridStats(std::optional<std::vector<std::string>> blocks_to_observe,
                   ::Events::ReductionScope reduction_scope)
      : blocks_to_observe_(std::move(blocks_to_observe)),
        reduction_scope_(reduction_scope) {}

  using observed_reduction_data_tags = observers::make_reduction_data_tags<
      tmpl::list<ObserveGridStats_detail::ReductionData>>;

  using compute_tags_for_observation_box = tmpl::list<NonTensorComputeTags...>;
  using argument_tags =
      tmpl::list<::Events::Tags::ObserverMesh<Dim>, domain::Tags::Domain<Dim>>;
  using return_tags = tmpl::list<>;

  template <typename Metavariables, typename ParallelComponent>
  void operator()(const Mesh<Dim>& observer_mesh, const Domain<Dim>& domain,
                  Parallel::GlobalCache<Metavariables>& cache,
                  const ElementId<Dim>& element_id,
                  const ParallelComponent* const /*meta*/,
                  const ObservationValue& observation_value) const {
    const std::optional<std::string> subfile_path =
        ObserveGridStats_detail::subfile_path(
            reduction_scope_, blocks_to_observe_, domain, element_id);
    if (not subfile_path.has_value()) {
      return;
    }
    auto& local_observer = *Parallel::local_branch(
        Parallel::get_parallel_component<
            tmpl::conditional_t<Parallel::is_nodegroup_v<ParallelComponent>,
                                observers::ObserverWriter<Metavariables>,
                                observers::Observer<Metavariables>>>(cache));
    observers::ObservationId observation_id{observation_value.value,
                                            subfile_path.value() + ".dat"};
    Parallel::ArrayComponentId array_component_id{
        std::add_pointer_t<ParallelComponent>{nullptr},
        Parallel::ArrayIndex<ElementId<Dim>>(element_id)};
    auto legend = ObserveGridStats_detail::legend(observation_value.name, Dim);
    auto reduction_data = ObserveGridStats_detail::reduction_data(
        observation_value.value, observer_mesh, element_id);
    // Chosing to only provide per-core granularity when reducing over entire
    // domain to avoid a subfile for every core x block combination
    const bool observe_per_core =
        reduction_scope_ == ::Events::ReductionScope::WholeDomain;
    if constexpr (Parallel::is_nodegroup_v<ParallelComponent>) {
      const std::optional<int> observe_with_core_id =
          observe_per_core
              ? std::make_optional(Parallel::my_node<int>(local_observer))
              : std::nullopt;
      Parallel::threaded_action<
          observers::ThreadedActions::CollectReductionDataOnNode>(
          local_observer, std::move(observation_id),
          std::move(array_component_id), subfile_path.value(),
          std::move(legend), std::move(reduction_data),
          std::optional<observers::NoFormatter>{}, observe_with_core_id);
    } else {
      Parallel::simple_action<observers::Actions::ContributeReductionData>(
          local_observer, std::move(observation_id),
          std::move(array_component_id), subfile_path.value(),
          std::move(legend), std::move(reduction_data),
          std::optional<observers::NoFormatter>{}, observe_per_core);
    }
  }

  using observation_registration_tags =
      tmpl::list<domain::Tags::Domain<Dim>, domain::Tags::Element<Dim>>;

  std::optional<
      std::pair<observers::TypeOfObservation, observers::ObservationKey>>
  get_observation_type_and_key_for_registration(
      const Domain<Dim>& domain, const Element<Dim>& element) const {
    const std::optional<std::string> subfile_path =
        ObserveGridStats_detail::subfile_path(
            reduction_scope_, blocks_to_observe_, domain, element.id());
    if (not subfile_path.has_value()) {
      return std::nullopt;
    }
    return {{observers::TypeOfObservation::Reduction,
             observers::ObservationKey(subfile_path.value() + ".dat")}};
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
  void pup(PUP::er& p) override {
    Event::pup(p);
    p | blocks_to_observe_;
    p | reduction_scope_;
  }

 private:
  std::optional<std::vector<std::string>> blocks_to_observe_{};
  ::Events::ReductionScope reduction_scope_{
      ::Events::ReductionScope::WholeDomain};
};

/// \cond
// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
template <size_t Dim, typename... NonTensorComputeTags>
PUP::able::PUP_ID
    ObserveGridStats<Dim, tmpl::list<NonTensorComputeTags...>>::my_PUP_ID = 0;
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)
/// \endcond
}  // namespace Events
