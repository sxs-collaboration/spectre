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
#include "Domain/InterfaceComputeTags.hpp"
#include "Domain/InterfaceHelpers.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/OrientationMapHelpers.hpp"
#include "Domain/Tags.hpp"
#include "IO/Observer/Actions/RegisterWithObservers.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "IO/Observer/TypeOfObservation.hpp"
#include "Informer/Tags.hpp"
#include "Informer/Verbosity.hpp"
#include "NumericalAlgorithms/Convergence/Tags.hpp"
#include "NumericalAlgorithms/LinearSolver/Gmres.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Options/Options.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InboxInserters.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/Printf.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/HasReceivedFromAllMortars.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Actions/CommunicateOverlapFields.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/ComputeTags.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/ElementCenteredSubdomainData.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/OverlapHelpers.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Protocols.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Tags.hpp"
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
    Parallel::ReductionDatum<size_t, funcl::Plus<>, funcl::Divides<>,
                             std::index_sequence<1>>,
    // Minimum number of subdomain solver iterations
    Parallel::ReductionDatum<size_t, funcl::Min<>>,
    // Maximum number of subdomain solver iterations
    Parallel::ReductionDatum<size_t, funcl::Max<>>>;

template <typename OptionsGroup>
struct RegisterObservers {
  template <typename ParallelComponent, typename DbTagsList,
            typename ArrayIndex>
  static std::pair<observers::TypeOfObservation, observers::ObservationKey>
  register_info(const db::DataBox<DbTagsList>& /*box*/,
                const ArrayIndex& /*array_index*/) noexcept {
    return {observers::TypeOfObservation::Reduction,
            observers::ObservationKey{pretty_type::get_name<OptionsGroup>() +
                                      "SubdomainSolves"}};
  }
};

template <typename FieldsTag, typename OptionsGroup, typename SourceTag>
using RegisterElement =
    observers::Actions::RegisterWithObservers<RegisterObservers<OptionsGroup>>;

template <typename OptionsGroup, typename ParallelComponent,
          typename Metavariables, typename ArrayIndex>
void contribute_to_subdomain_stats_observation(
    const size_t iteration_id, const size_t subdomain_solve_num_iterations,
    Parallel::GlobalCache<Metavariables>& cache,
    const ArrayIndex& array_index) noexcept {
  auto& local_observer =
      *Parallel::get_parallel_component<observers::Observer<Metavariables>>(
           cache)
           .ckLocalBranch();
  Parallel::simple_action<observers::Actions::ContributeReductionData>(
      local_observer,
      observers::ObservationId(
          iteration_id,
          pretty_type::get_name<OptionsGroup>() + "SubdomainSolves"),
      observers::ArrayComponentId{
          std::add_pointer_t<ParallelComponent>{nullptr},
          Parallel::ArrayIndex<ArrayIndex>(array_index)},
      std::string{"/" + Options::name<OptionsGroup>() + "SubdomainSolves"},
      std::vector<std::string>{"Iteration", "NumSubdomains", "AvgNumIterations",
                               "MinNumIterations", "MaxNumIterations"},
      reduction_data{iteration_id, 1, subdomain_solve_num_iterations,
                     subdomain_solve_num_iterations,
                     subdomain_solve_num_iterations});
}

template <typename SubdomainDataType, typename OptionsGroup>
struct SubdomainDataBufferTag : db::SimpleTag {
  static std::string name() noexcept {
    return "SubdomainData(" + Options::name<OptionsGroup>() + ")";
  }
  using type = SubdomainDataType;
};

template <typename FieldsTag, typename OptionsGroup, typename SubdomainOperator>
struct InitializeElement {
 private:
  using fields_tag = FieldsTag;
  using residual_tag =
      db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>;
  static constexpr size_t Dim = SubdomainOperator::volume_dim;
  using SubdomainData =
      ElementCenteredSubdomainData<Dim, typename residual_tag::tags_list>;
  // Here we choose a serial GMRES linear solver to solve subdomain problems.
  // This can be generalized to allow the user to make this choice once that
  // becomes necessary.
  using subdomain_solver_tag =
      Tags::SubdomainSolver<LinearSolver::Serial::Gmres<SubdomainData>,
                            OptionsGroup>;

 public:
  using initialization_tags =
      tmpl::list<domain::Tags::InitialExtents<Dim>, subdomain_solver_tag>;
  using initialization_tags_to_keep = tmpl::list<subdomain_solver_tag>;
  using const_global_cache_tags = tmpl::list<Tags::MaxOverlap<OptionsGroup>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ElementId<Dim>& /*element_id*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    const auto& element_mesh = db::get<domain::Tags::Mesh<Dim>>(box);
    const size_t element_num_points = element_mesh.number_of_grid_points();
    return std::make_tuple(
        ::Initialization::merge_into_databox<
            InitializeElement,
            db::AddSimpleTags<
                Tags::Overlaps<residual_tag, Dim, OptionsGroup>,
                SubdomainDataBufferTag<SubdomainData, OptionsGroup>>,
            db::AddComputeTags<
                domain::Tags::InternalDirectionsCompute<Dim>,
                domain::Tags::BoundaryDirectionsInteriorCompute<Dim>,
                domain::Tags::InterfaceCompute<
                    domain::Tags::InternalDirections<Dim>,
                    domain::Tags::Direction<Dim>>,
                domain::Tags::InterfaceCompute<
                    domain::Tags::BoundaryDirectionsInterior<Dim>,
                    domain::Tags::Direction<Dim>>,
                Tags::IntrudingExtentsCompute<Dim, OptionsGroup>,
                Tags::IntrudingOverlapWidthsCompute<Dim, OptionsGroup>,
                domain::Tags::LogicalCoordinates<Dim>,
                Tags::ElementWeightCompute<Dim, OptionsGroup>,
                domain::Tags::InterfaceCompute<
                    domain::Tags::InternalDirections<Dim>,
                    Tags::IntrudingOverlapWeightCompute<Dim, OptionsGroup>>>>(
            std::move(box),
            typename Tags::Overlaps<residual_tag, Dim, OptionsGroup>::type{},
            SubdomainData{element_num_points}));
  }
};

// Restrict the residual to neighboring subdomains that overlap with this
// element and send the data to those elements
template <typename FieldsTag, typename OptionsGroup, typename SubdomainOperator>
using SendOverlapData = LinearSolver::Schwarz::Actions::SendOverlapFields<
    tmpl::list<db::add_tag_prefix<LinearSolver::Tags::Residual, FieldsTag>>,
    OptionsGroup, true>;

// Wait for the residual data on regions of this element's subdomain that
// overlap with other elements.
template <typename FieldsTag, typename OptionsGroup, typename SubdomainOperator>
using ReceiveOverlapData = LinearSolver::Schwarz::Actions::ReceiveOverlapFields<
    SubdomainOperator::volume_dim,
    tmpl::list<db::add_tag_prefix<LinearSolver::Tags::Residual, FieldsTag>>,
    OptionsGroup>;

template <size_t Dim, typename OptionsGroup, typename OverlapSolution>
struct OverlapSolutionInboxTag
    : public Parallel::InboxInserters::Map<
          OverlapSolutionInboxTag<Dim, OptionsGroup, OverlapSolution>> {
  using temporal_id = size_t;
  using type = std::map<temporal_id, OverlapMap<Dim, OverlapSolution>>;
};

// Once the residual data is available on all overlaps, solve the restricted
// problem for this element-centered subdomain. Apply the weighted solution on
// this element directly and send the solution on overlap regions to the
// neighbors that they overlap with.
template <typename FieldsTag, typename OptionsGroup, typename SubdomainOperator>
struct SolveSubdomain {
 private:
  using fields_tag = FieldsTag;
  using residual_tag =
      db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>;
  static constexpr size_t Dim = SubdomainOperator::volume_dim;
  using SubdomainData =
      ElementCenteredSubdomainData<Dim, typename residual_tag::tags_list>;
  using OverlapData = typename SubdomainData::OverlapData;
  using overlap_solution_inbox_tag =
      OverlapSolutionInboxTag<Dim, OptionsGroup, OverlapData>;

 public:
  using const_global_cache_tags =
      tmpl::list<Tags::MaxOverlap<OptionsGroup>,
                 logging::Tags::Verbosity<OptionsGroup>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ElementId<Dim>& element_id, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    const size_t iteration_id =
        get<Convergence::Tags::IterationId<OptionsGroup>>(box);

    // Do some logging
    if (UNLIKELY(get<logging::Tags::Verbosity<OptionsGroup>>(box) >=
                 ::Verbosity::Debug)) {
      Parallel::printf(
          "%s " + Options::name<OptionsGroup>() + "(%zu): Solve subdomain\n",
          element_id, iteration_id);
    }

    const auto& element = db::get<domain::Tags::Element<Dim>>(box);
    const size_t max_overlap = db::get<Tags::MaxOverlap<OptionsGroup>>(box);

    // Assemble the subdomain data from the data on the element and the
    // communicated overlap data
    db::mutate<SubdomainDataBufferTag<SubdomainData, OptionsGroup>,
               Tags::Overlaps<residual_tag, Dim, OptionsGroup>>(
        make_not_null(&box),
        [max_overlap, &element](
            const gsl::not_null<SubdomainData*> subdomain_data,
            const auto overlap_residuals, const auto& residual) noexcept {
          subdomain_data->element_data = residual;
          // Nothing was communicated if the overlaps are empty
          if (LIKELY(max_overlap > 0 and element.number_of_neighbors() > 0)) {
            subdomain_data->overlap_data = std::move(*overlap_residuals);
          }
        },
        db::get<residual_tag>(box));
    const auto& subdomain_residual =
        db::get<SubdomainDataBufferTag<SubdomainData, OptionsGroup>>(box);

    // Allocate workspace memory for repeatedly applying the subdomain operator
    const size_t num_points =
        db::get<domain::Tags::Mesh<Dim>>(box).number_of_grid_points();
    SubdomainOperator subdomain_operator{num_points};
    auto subdomain_result_buffer =
        make_with_value<SubdomainData>(subdomain_residual, 0.);

    // Construct the subdomain operator
    const auto apply_subdomain_operator = [&box, &subdomain_result_buffer,
                                           &subdomain_operator](
                                              const SubdomainData&
                                                  arg) noexcept {
      // The subdomain operator can retrieve any information on the subdomain
      // geometry that is available through the DataBox. The user is responsible
      // for communicating this information across neighbors if necessary.
      db::apply<typename SubdomainOperator::element_operator>(
          box, arg, make_not_null(&subdomain_result_buffer),
          make_not_null(&subdomain_operator));
      tmpl::for_each<tmpl::list<domain::Tags::InternalDirections<Dim>,
                                domain::Tags::BoundaryDirectionsInterior<Dim>>>(
          [&box, &arg, &subdomain_result_buffer,
           &subdomain_operator](auto directions_v) noexcept {
            using directions = tmpl::type_from<decltype(directions_v)>;
            using face_operator =
                typename SubdomainOperator::template face_operator<directions>;
            interface_apply<directions, face_operator>(
                box, arg, make_not_null(&subdomain_result_buffer),
                make_not_null(&subdomain_operator));
          });
      return subdomain_result_buffer;
    };

    // Solve the subdomain problem
    const auto& subdomain_solver =
        get<Tags::SubdomainSolverBase<OptionsGroup>>(box);
    Convergence::HasConverged subdomain_solve_has_converged{};
    std::tie(subdomain_solve_has_converged, subdomain_result_buffer) =
        subdomain_solver(
            apply_subdomain_operator, subdomain_residual,
            make_with_value<SubdomainData>(subdomain_residual, 0.));
    // We're re-using the buffer to store the subdomain solution. Re-naming it
    // here for the code below.
    auto& subdomain_solution = subdomain_result_buffer;

    // Do some logging and observing
    if (UNLIKELY(get<logging::Tags::Verbosity<OptionsGroup>>(box) >=
                 ::Verbosity::Quiet)) {
      if (not subdomain_solve_has_converged or
          subdomain_solve_has_converged.reason() ==
              Convergence::Reason::MaxIterations) {
        Parallel::printf(
            "%s WARNING: Subdomain solver did not converge in %zu iterations: "
            "%e -> %e\n",
            element_id, subdomain_solve_has_converged.num_iterations(),
            subdomain_solve_has_converged.initial_residual_magnitude(),
            subdomain_solve_has_converged.residual_magnitude());
      } else if (UNLIKELY(get<logging::Tags::Verbosity<OptionsGroup>>(box) >=
                          ::Verbosity::Debug)) {
        Parallel::printf(
            "%s Subdomain solver converged in %zu iterations (%s): %e -> %e\n",
            element_id, subdomain_solve_has_converged.num_iterations(),
            subdomain_solve_has_converged.reason(),
            subdomain_solve_has_converged.initial_residual_magnitude(),
            subdomain_solve_has_converged.residual_magnitude());
      }
    }
    contribute_to_subdomain_stats_observation<OptionsGroup, ParallelComponent>(
        iteration_id + 1, subdomain_solve_has_converged.num_iterations(), cache,
        element_id);

    // Apply weighting
    if (LIKELY(max_overlap > 0)) {
      subdomain_solution.element_data *=
          get(db::get<Tags::Weight<OptionsGroup>>(box));
    }

    // Apply solution to central element
    db::mutate<fields_tag>(make_not_null(&box),
                           [&subdomain_solution](const auto fields) noexcept {
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
    return {std::move(box)};
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
            size_t Dim>
  static bool is_ready(const db::DataBox<DbTagsList>& box,
                       const tuples::TaggedTuple<InboxTags...>& inboxes,
                       const Parallel::GlobalCache<Metavariables>& /*cache*/,
                       const ElementId<Dim>& /*element_id*/) noexcept {
    if (UNLIKELY(db::get<Tags::MaxOverlap<OptionsGroup>>(box) == 0)) {
      return true;
    }
    return dg::has_received_from_all_mortars<overlap_solution_inbox_tag>(
        get<Convergence::Tags::IterationId<OptionsGroup>>(box),
        get<domain::Tags::Element<Dim>>(box), inboxes);
  }

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& element_id, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    const size_t iteration_id =
        get<Convergence::Tags::IterationId<OptionsGroup>>(box);
    const auto& element = db::get<domain::Tags::Element<Dim>>(box);

    // Nothing to do if overlap is empty
    if (UNLIKELY(db::get<Tags::MaxOverlap<OptionsGroup>>(box) == 0 or
                 element.number_of_neighbors() == 0)) {
      return {std::move(box)};
    }

    // Do some logging
    if (UNLIKELY(get<logging::Tags::Verbosity<OptionsGroup>>(box) >=
                 ::Verbosity::Debug)) {
      Parallel::printf("%s " + Options::name<OptionsGroup>() +
                           "(%zu): Receive overlap solution\n",
                       element_id, iteration_id);
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
            const std::unordered_map<Direction<Dim>, Scalar<DataVector>>&
                all_intruding_overlap_weights) noexcept {
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
        db::get<domain::Tags::Interface<domain::Tags::InternalDirections<Dim>,
                                        Tags::Weight<OptionsGroup>>>(box));
    return {std::move(box)};
  }
};

}  // namespace LinearSolver::Schwarz::detail
