// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <map>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Index.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/OrientationMapHelpers.hpp"
#include "Domain/Tags.hpp"
#include "Domain/Tags/Faces.hpp"
#include "IO/Logging/Tags.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "IO/Observer/Actions/RegisterWithObservers.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/GetSectionObservationKey.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/Protocols/ReductionDataFormatter.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "IO/Observer/Tags.hpp"
#include "IO/Observer/TypeOfObservation.hpp"
#include "NumericalAlgorithms/Convergence/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/HasReceivedFromAllMortars.hpp"
#include "NumericalAlgorithms/LinearSolver/ExplicitInverse.hpp"
#include "NumericalAlgorithms/LinearSolver/Gmres.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Options/Options.hpp"
#include "Parallel/AlgorithmMetafunctions.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InboxInserters.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/Printf.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "ParallelAlgorithms/LinearSolver/AsynchronousSolvers/ElementActions.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Actions/CommunicateOverlapFields.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/ElementCenteredSubdomainData.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/OverlapHelpers.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Tags.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Weighting.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace LinearSolver::Schwarz::detail {

using reduction_data = Parallel::ReductionData<
    // Iteration
    Parallel::ReductionDatum<size_t, funcl::AssertEqual<>>,
    // Number of subdomains (= number of elements)
    Parallel::ReductionDatum<size_t, funcl::Plus<>>,
    // Average number of subdomain solver iterations
    Parallel::ReductionDatum<double, funcl::Plus<>, funcl::Divides<>,
                             std::index_sequence<1>>,
    // Minimum number of subdomain solver iterations
    Parallel::ReductionDatum<size_t, funcl::Min<>>,
    // Maximum number of subdomain solver iterations
    Parallel::ReductionDatum<size_t, funcl::Max<>>,
    // Total number of subdomain solver iterations
    Parallel::ReductionDatum<size_t, funcl::Plus<>>>;

template <typename OptionsGroup>
struct SubdomainStatsFormatter
    : tt::ConformsTo<observers::protocols::ReductionDataFormatter> {
  using reduction_data = Schwarz::detail::reduction_data;
  SubdomainStatsFormatter() = default;
  SubdomainStatsFormatter(std::string local_section_observation_key)
      : section_observation_key(std::move(local_section_observation_key)) {}
  std::string operator()(const size_t iteration_id, const size_t num_subdomains,
                         const double avg_subdomain_its,
                         const size_t min_subdomain_its,
                         const size_t max_subdomain_its,
                         const size_t total_subdomain_its) const {
    return Options::name<OptionsGroup>() + section_observation_key + "(" +
           get_output(iteration_id) + ") completed all " +
           get_output(num_subdomains) +
           " subdomain solves. Average of number of iterations: " +
           get_output(avg_subdomain_its) + " (min " +
           get_output(min_subdomain_its) + ", max " +
           get_output(max_subdomain_its) + ", total " +
           get_output(total_subdomain_its) + ").";
  }
  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) { p | section_observation_key; }
  std::string section_observation_key{};
};

template <typename OptionsGroup, typename ArraySectionIdTag>
struct RegisterObservers {
  template <typename ParallelComponent, typename DbTagsList,
            typename ArrayIndex>
  static std::pair<observers::TypeOfObservation, observers::ObservationKey>
  register_info(const db::DataBox<DbTagsList>& box,
                const ArrayIndex& /*array_index*/) {
    // Get the observation key, or "Unused" if the element does not belong
    // to a section with this tag. In the latter case, no observations will
    // ever be contributed.
    const std::optional<std::string> section_observation_key =
        observers::get_section_observation_key<ArraySectionIdTag>(box);
    ASSERT(section_observation_key != "Unused",
           "The identifier 'Unused' is reserved to indicate that no "
           "observations with this key will be contributed. Use a different "
           "key, or change the identifier 'Unused' to something else.");
    return {
        observers::TypeOfObservation::Reduction,
        observers::ObservationKey{pretty_type::get_name<OptionsGroup>() +
                                  section_observation_key.value_or("Unused") +
                                  "SubdomainSolves"}};
  }
};

template <typename FieldsTag, typename OptionsGroup, typename SourceTag,
          typename ArraySectionIdTag>
using RegisterElement = observers::Actions::RegisterWithObservers<
    RegisterObservers<OptionsGroup, ArraySectionIdTag>>;

template <typename OptionsGroup, typename ParallelComponent,
          typename Metavariables, typename ArrayIndex>
void contribute_to_subdomain_stats_observation(
    const size_t iteration_id, const size_t subdomain_solve_num_iterations,
    Parallel::GlobalCache<Metavariables>& cache, const ArrayIndex& array_index,
    const std::string& section_observation_key, const bool observe_per_core) {
  auto& local_observer =
      *Parallel::get_parallel_component<observers::Observer<Metavariables>>(
           cache)
           .ckLocalBranch();
  auto formatter =
      UNLIKELY(get<logging::Tags::Verbosity<OptionsGroup>>(cache) >=
               ::Verbosity::Verbose)
          ? std::make_optional(
                SubdomainStatsFormatter<OptionsGroup>{section_observation_key})
          : std::nullopt;
  Parallel::simple_action<observers::Actions::ContributeReductionData>(
      local_observer,
      observers::ObservationId(iteration_id,
                               pretty_type::get_name<OptionsGroup>() +
                                   section_observation_key + "SubdomainSolves"),
      observers::ArrayComponentId{
          std::add_pointer_t<ParallelComponent>{nullptr},
          Parallel::ArrayIndex<ArrayIndex>(array_index)},
      std::string{"/" + Options::name<OptionsGroup>() +
                  section_observation_key + "SubdomainSolves"},
      std::vector<std::string>{"Iteration", "NumSubdomains", "AvgNumIterations",
                               "MinNumIterations", "MaxNumIterations",
                               "TotalNumIterations"},
      reduction_data{
          iteration_id, 1, static_cast<double>(subdomain_solve_num_iterations),
          subdomain_solve_num_iterations, subdomain_solve_num_iterations,
          subdomain_solve_num_iterations},
      std::move(formatter), observe_per_core);
}

template <typename SubdomainDataType, typename OptionsGroup>
struct SubdomainDataBufferTag : db::SimpleTag {
  static std::string name() {
    return "SubdomainData(" + Options::name<OptionsGroup>() + ")";
  }
  using type = SubdomainDataType;
};

// Allow factory-creating any of these serial linear solvers for use as
// subdomain solver
template <typename FieldsTag, typename SubdomainOperator,
          typename SubdomainPreconditioners,
          typename SubdomainData = ElementCenteredSubdomainData<
              SubdomainOperator::volume_dim,
              typename db::add_tag_prefix<LinearSolver::Tags::Residual,
                                          FieldsTag>::tags_list>>
using subdomain_solver = LinearSolver::Serial::LinearSolver<tmpl::append<
    tmpl::list<::LinearSolver::Serial::Registrars::Gmres<SubdomainData>,
               ::LinearSolver::Serial::Registrars::ExplicitInverse>,
    SubdomainPreconditioners>>;

template <typename FieldsTag, typename OptionsGroup, typename SubdomainOperator,
          typename SubdomainPreconditioners>
struct InitializeElement {
 private:
  using fields_tag = FieldsTag;
  using residual_tag =
      db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>;
  static constexpr size_t Dim = SubdomainOperator::volume_dim;
  using SubdomainData =
      ElementCenteredSubdomainData<Dim, typename residual_tag::tags_list>;
  using subdomain_solver_tag = Tags::SubdomainSolver<
      std::unique_ptr<subdomain_solver<FieldsTag, SubdomainOperator,
                                       SubdomainPreconditioners>>,
      OptionsGroup>;

 public:
  using initialization_tags =
      tmpl::list<domain::Tags::InitialExtents<Dim>, subdomain_solver_tag>;
  using initialization_tags_to_keep = tmpl::list<subdomain_solver_tag>;
  using const_global_cache_tags = tmpl::list<Tags::MaxOverlap<OptionsGroup>>;

  using simple_tags =
      tmpl::list<Tags::IntrudingExtents<Dim, OptionsGroup>,
                 Tags::Weight<OptionsGroup>,
                 domain::Tags::Faces<Dim, Tags::Weight<OptionsGroup>>,
                 SubdomainDataBufferTag<SubdomainData, OptionsGroup>>;
  using compute_tags = tmpl::list<>;
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ElementId<Dim>& /*element_id*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) {
    const auto& element = db::get<domain::Tags::Element<Dim>>(box);
    const auto& mesh = db::get<domain::Tags::Mesh<Dim>>(box);
    const size_t num_points = mesh.number_of_grid_points();
    const auto& logical_coords =
        db::get<domain::Tags::Coordinates<Dim, Frame::ElementLogical>>(box);
    const size_t max_overlap = db::get<Tags::MaxOverlap<OptionsGroup>>(box);

    // Intruding overlaps
    std::array<size_t, Dim> intruding_extents{};
    std::array<double, Dim> intruding_overlap_widths{};
    for (size_t d = 0; d < Dim; ++d) {
      gsl::at(intruding_extents, d) =
          LinearSolver::Schwarz::overlap_extent(mesh.extents(d), max_overlap);
      const auto& collocation_points =
          Spectral::collocation_points(mesh.slice_through(d));
      gsl::at(intruding_overlap_widths, d) =
          LinearSolver::Schwarz::overlap_width(gsl::at(intruding_extents, d),
                                               collocation_points);
    }

    // Element weight
    Scalar<DataVector> element_weight{num_points, 1.};
    // For max_overlap > 0 all overlaps will have non-zero extents on an LGL
    // mesh (because it has at least 2 points per dimension), so we don't need
    // to check their extents are non-zero individually
    if (LIKELY(max_overlap > 0)) {
      LinearSolver::Schwarz::element_weight(
          make_not_null(&element_weight), logical_coords,
          intruding_overlap_widths, element.external_boundaries());
    }

    // Intruding overlap weights
    DirectionMap<Dim, Scalar<DataVector>> intruding_overlap_weights{};
    for (const auto& direction : element.internal_boundaries()) {
      const size_t dim = direction.dimension();
      if (gsl::at(intruding_extents, dim) > 0) {
        const auto intruding_logical_coords =
            LinearSolver::Schwarz::data_on_overlap(
                logical_coords, mesh.extents(), gsl::at(intruding_extents, dim),
                direction);
        intruding_overlap_weights[direction] =
            LinearSolver::Schwarz::intruding_weight(
                intruding_logical_coords, direction, intruding_overlap_widths,
                element.neighbors().at(direction).size(),
                element.external_boundaries());
      }
    }

    Initialization::mutate_assign<simple_tags>(
        make_not_null(&box), std::move(intruding_extents),
        std::move(element_weight), std::move(intruding_overlap_weights),
        SubdomainData{num_points});
    return std::make_tuple(std::move(box));
  }
};

// Restrict the residual to neighboring subdomains that overlap with this
// element and send the data to those elements
template <typename FieldsTag, typename OptionsGroup, typename SubdomainOperator>
using SendOverlapData = LinearSolver::Schwarz::Actions::SendOverlapFields<
    tmpl::list<db::add_tag_prefix<LinearSolver::Tags::Residual, FieldsTag>>,
    OptionsGroup, true>;

template <size_t Dim, typename OptionsGroup, typename OverlapSolution>
struct OverlapSolutionInboxTag
    : public Parallel::InboxInserters::Map<
          OverlapSolutionInboxTag<Dim, OptionsGroup, OverlapSolution>> {
  using temporal_id = size_t;
  using type = std::map<temporal_id, OverlapMap<Dim, OverlapSolution>>;
};

// Wait for the residual data on regions of this element's subdomain that
// overlap with other elements. Once the residual data is available on all
// overlaps, solve the restricted problem for this element-centered subdomain.
// Apply the weighted solution on this element directly and send the solution on
// overlap regions to the neighbors that they overlap with.
template <typename FieldsTag, typename OptionsGroup, typename SubdomainOperator,
          typename ArraySectionIdTag>
struct SolveSubdomain {
 private:
  using fields_tag = FieldsTag;
  using residual_tag =
      db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>;
  static constexpr size_t Dim = SubdomainOperator::volume_dim;
  using overlap_residuals_inbox_tag =
      Actions::detail::OverlapFieldsTag<Dim, tmpl::list<residual_tag>,
                                        OptionsGroup>;
  using SubdomainData =
      ElementCenteredSubdomainData<Dim, typename residual_tag::tags_list>;
  using OverlapData = typename SubdomainData::OverlapData;
  using overlap_solution_inbox_tag =
      OverlapSolutionInboxTag<Dim, OptionsGroup, OverlapData>;

 public:
  using const_global_cache_tags =
      tmpl::list<Tags::MaxOverlap<OptionsGroup>,
                 logging::Tags::Verbosity<OptionsGroup>,
                 Tags::ObservePerCoreReductions<OptionsGroup>>;
  using inbox_tags = tmpl::list<overlap_residuals_inbox_tag>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, Parallel::AlgorithmExecution>
  apply(db::DataBox<DbTagsList>& box,
        tuples::TaggedTuple<InboxTags...>& inboxes,
        Parallel::GlobalCache<Metavariables>& cache,
        const ElementId<Dim>& element_id, const ActionList /*meta*/,
        const ParallelComponent* const /*meta*/) {
    const size_t iteration_id =
        get<Convergence::Tags::IterationId<OptionsGroup>>(box);
    const auto& element = db::get<domain::Tags::Element<Dim>>(box);
    const size_t max_overlap = db::get<Tags::MaxOverlap<OptionsGroup>>(box);

    // Wait for communicated overlap data
    const bool has_overlap_data =
        max_overlap > 0 and element.number_of_neighbors() > 0;
    if (LIKELY(has_overlap_data) and
        not dg::has_received_from_all_mortars<overlap_residuals_inbox_tag>(
            iteration_id, element, inboxes)) {
      return {std::move(box), Parallel::AlgorithmExecution::Retry};
    }

    // Do some logging
    if (UNLIKELY(get<logging::Tags::Verbosity<OptionsGroup>>(box) >=
                 ::Verbosity::Debug)) {
      Parallel::printf("%s %s(%zu): Solve subdomain\n", element_id,
                       Options::name<OptionsGroup>(), iteration_id);
    }

    // Assemble the subdomain data from the data on the element and the
    // communicated overlap data
    db::mutate<SubdomainDataBufferTag<SubdomainData, OptionsGroup>>(
        make_not_null(&box),
        [&inboxes, &iteration_id, &has_overlap_data](
            const gsl::not_null<SubdomainData*> subdomain_data,
            const auto& residual) {
          subdomain_data->element_data = residual;
          // Nothing was communicated if the overlaps are empty
          if (LIKELY(has_overlap_data)) {
            subdomain_data->overlap_data =
                std::move(tuples::get<overlap_residuals_inbox_tag>(inboxes)
                              .extract(iteration_id)
                              .mapped());
          }
        },
        db::get<residual_tag>(box));
    const auto& subdomain_residual =
        db::get<SubdomainDataBufferTag<SubdomainData, OptionsGroup>>(box);

    // Allocate workspace memory for repeatedly applying the subdomain operator
    const SubdomainOperator subdomain_operator{};

    // Solve the subdomain problem
    const auto& subdomain_solver =
        get<Tags::SubdomainSolverBase<OptionsGroup>>(box);
    auto subdomain_solve_initial_guess_in_solution_out =
        make_with_value<SubdomainData>(subdomain_residual, 0.);
    const auto subdomain_solve_has_converged = subdomain_solver.solve(
        make_not_null(&subdomain_solve_initial_guess_in_solution_out),
        subdomain_operator, subdomain_residual, std::forward_as_tuple(box));
    // Re-naming the solution buffer for the code below
    auto& subdomain_solution = subdomain_solve_initial_guess_in_solution_out;

    // Do some logging and observing
    if (UNLIKELY(get<logging::Tags::Verbosity<OptionsGroup>>(box) >=
                 ::Verbosity::Quiet)) {
      if (not subdomain_solve_has_converged or
          subdomain_solve_has_converged.reason() ==
              Convergence::Reason::MaxIterations) {
        Parallel::printf(
            "%s %s(%zu): WARNING: Subdomain solver did not converge in %zu "
            "iterations: %e -> %e\n",
            element_id, Options::name<OptionsGroup>(), iteration_id,
            subdomain_solve_has_converged.num_iterations(),
            subdomain_solve_has_converged.initial_residual_magnitude(),
            subdomain_solve_has_converged.residual_magnitude());
      } else if (UNLIKELY(get<logging::Tags::Verbosity<OptionsGroup>>(box) >=
                          ::Verbosity::Debug)) {
        Parallel::printf(
            "%s %s(%zu): Subdomain solver converged in %zu iterations (%s): %e "
            "-> %e\n",
            element_id, Options::name<OptionsGroup>(), iteration_id,
            subdomain_solve_has_converged.num_iterations(),
            subdomain_solve_has_converged.reason(),
            subdomain_solve_has_converged.initial_residual_magnitude(),
            subdomain_solve_has_converged.residual_magnitude());
      }
    }
    const std::optional<std::string> section_observation_key =
        observers::get_section_observation_key<ArraySectionIdTag>(box);
    if (section_observation_key.has_value()) {
      contribute_to_subdomain_stats_observation<OptionsGroup,
                                                ParallelComponent>(
          iteration_id + 1, subdomain_solve_has_converged.num_iterations(),
          cache, element_id, *section_observation_key,
          db::get<Tags::ObservePerCoreReductions<OptionsGroup>>(box));
    }

    // Apply weighting
    if (LIKELY(max_overlap > 0)) {
      subdomain_solution.element_data *=
          get(db::get<Tags::Weight<OptionsGroup>>(box));
    }

    // Apply solution to central element
    db::mutate<fields_tag>(make_not_null(&box),
                           [&subdomain_solution](const auto fields) {
                             *fields += subdomain_solution.element_data;
                           });

    // Send overlap solutions back to the neighbors that they are on
    if (LIKELY(max_overlap > 0)) {
      auto& receiver_proxy =
          Parallel::get_parallel_component<ParallelComponent>(cache);
      for (auto& [overlap_id, overlap_solution] :
           subdomain_solution.overlap_data) {
        const auto& direction = overlap_id.first;
        const auto& neighbor_id = overlap_id.second;
        const auto& orientation =
            element.neighbors().at(direction).orientation();
        const auto direction_from_neighbor = orientation(direction.opposite());
        Parallel::receive_data<overlap_solution_inbox_tag>(
            receiver_proxy[neighbor_id], iteration_id,
            std::make_pair(
                OverlapId<Dim>{direction_from_neighbor, element.id()},
                std::move(overlap_solution)));
      }
    }
    return {std::move(box), Parallel::AlgorithmExecution::Continue};
  }
};

// Wait for the subdomain solutions on regions within this element that overlap
// with neighboring element-centered subdomains. Combine the solutions as a
// weighted sum.
template <typename FieldsTag, typename OptionsGroup, typename SubdomainOperator>
struct ReceiveOverlapSolution {
 private:
  using fields_tag = FieldsTag;
  using residual_tag =
      db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>;
  static constexpr size_t Dim = SubdomainOperator::volume_dim;
  using SubdomainData =
      ElementCenteredSubdomainData<Dim, typename residual_tag::tags_list>;
  using OverlapSolution = typename SubdomainData::OverlapData;
  using overlap_solution_inbox_tag =
      OverlapSolutionInboxTag<Dim, OptionsGroup, OverlapSolution>;

 public:
  using const_global_cache_tags = tmpl::list<Tags::MaxOverlap<OptionsGroup>>;
  using inbox_tags = tmpl::list<overlap_solution_inbox_tag>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, Parallel::AlgorithmExecution>
  apply(db::DataBox<DbTagsList>& box,
        tuples::TaggedTuple<InboxTags...>& inboxes,
        const Parallel::GlobalCache<Metavariables>& /*cache*/,
        const ElementId<Dim>& element_id, const ActionList /*meta*/,
        const ParallelComponent* const /*meta*/) {
    const size_t iteration_id =
        get<Convergence::Tags::IterationId<OptionsGroup>>(box);
    const auto& element = db::get<domain::Tags::Element<Dim>>(box);

    // Nothing to do if overlap is empty
    if (UNLIKELY(db::get<Tags::MaxOverlap<OptionsGroup>>(box) == 0 or
                 element.number_of_neighbors() == 0)) {
      return {std::move(box), Parallel::AlgorithmExecution::Continue};
    }

    if (not dg::has_received_from_all_mortars<overlap_solution_inbox_tag>(
            iteration_id, element, inboxes)) {
      return {std::move(box), Parallel::AlgorithmExecution::Retry};
    }

    // Do some logging
    if (UNLIKELY(get<logging::Tags::Verbosity<OptionsGroup>>(box) >=
                 ::Verbosity::Debug)) {
      Parallel::printf("%s %s(%zu): Receive overlap solution\n", element_id,
                       Options::name<OptionsGroup>(), iteration_id);
    }

    // Add solutions on overlaps to this element's solution in a weighted sum
    const auto received_overlap_solutions =
        std::move(tuples::get<overlap_solution_inbox_tag>(inboxes)
                      .extract(iteration_id)
                      .mapped());
    db::mutate<fields_tag>(
        make_not_null(&box),
        [&received_overlap_solutions](
            const auto fields, const Index<Dim>& full_extents,
            const std::array<size_t, Dim>& all_intruding_extents,
            const DirectionMap<Dim, Scalar<DataVector>>&
                all_intruding_overlap_weights) {
          for (const auto& [overlap_id, overlap_solution] :
               received_overlap_solutions) {
            const auto& direction = overlap_id.first;
            const auto& intruding_extents =
                gsl::at(all_intruding_extents, direction.dimension());
            const auto& overlap_weight =
                all_intruding_overlap_weights.at(direction);
            LinearSolver::Schwarz::add_overlap_data(
                fields, overlap_solution * get(overlap_weight), full_extents,
                intruding_extents, direction);
          }
        },
        db::get<domain::Tags::Mesh<Dim>>(box).extents(),
        db::get<Tags::IntrudingExtents<Dim, OptionsGroup>>(box),
        db::get<domain::Tags::Faces<Dim, Tags::Weight<OptionsGroup>>>(box));
    return {std::move(box), Parallel::AlgorithmExecution::Continue};
  }
};

}  // namespace LinearSolver::Schwarz::detail
