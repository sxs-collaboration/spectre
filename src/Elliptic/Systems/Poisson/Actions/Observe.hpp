// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Systems/Poisson/Tags.hpp"
#include "IO/Observer/Actions.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "IO/Observer/VolumeActions.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Reduction.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Poisson {
namespace Actions {

/*!
 * \brief Action to observe volume and reduction data for the Poisson system
 *
 * This action observes the following:
 * - Reduction data:
 *   - `"Iteration"`: The linear solver iteration
 *   - `"NumberOfPoints"`: The total number of grid points across all elements
 *   - `"L2Error"`: The standard L2 vector norm of the pointwise difference
 *     between the numerical and analytic solution \f$\sqrt{\sum_i \left(
 *     u^\mathrm{numerical}_i - u^\mathrm{analytic}_i\right)^2}\f$ over the grid
 *     points across all elements.
 * - Volume data:
 *   - `Poisson::Field::name()`: The numerical solution
 *     \f$u^\mathrm{numerical}\f$
 *   - `Poisson::Field::name() + "Analytic"`: The analytic solution
 *     \f$u^\mathrm{analytic}\f$
 *   - `Poisson::Field::name() + "Error"`: The pointwise error
 *     \f$u^\mathrm{numerical} - u^\mathrm{analytic}\f$
 *   - `"InertialCoordinates_{x,y,z}"`
 *
 * Uses:
 * - Metavariables:
 *   - `analytic_solution_tag`
 *   - Items required by `observers::Observer<Metavariables>`
 * - DataBox:
 *   - `LinearSolver::Tags::IterationId`
 *   - `Tags::Mesh<Dim>`
 *   - `Poisson::Field`
 *   - `Tags::Coordinates<Dim, Frame::Inertial>`
 *
 * \note This action can be adjusted before compiling an executable to observe
 * only the desired quantities.
 */
struct Observe {
 private:
  using observed_reduction_data = Parallel::ReductionData<
      Parallel::ReductionDatum<size_t, funcl::AssertEqual<>>,
      Parallel::ReductionDatum<size_t, funcl::Plus<>>,
      Parallel::ReductionDatum<double, funcl::Plus<>,
                               funcl::Sqrt<funcl::Divides<>>,
                               std::index_sequence<1>>>;

 public:
  // Compile-time interface for observers
  using observed_reduction_data_tags =
      observers::make_reduction_data_tags<tmpl::list<observed_reduction_data>>;

  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static auto apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ElementIndex<Dim>& array_index,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    const auto& iteration_id = get<LinearSolver::Tags::IterationId>(box);
    const auto& mesh = get<Tags::Mesh<Dim>>(box);
    const std::string element_name = MakeString{} << ElementId<Dim>(array_index)
                                                  << '/';

    // Retrieve the current numeric solution
    const auto& field = get<Poisson::Field>(box);

    // Compute the analytic solution
    const auto& inertial_coordinates =
        db::get<::Tags::Coordinates<Dim, Frame::Inertial>>(box);
    const auto field_analytic = get<Poisson::Field>(
        Parallel::get<typename Metavariables::analytic_solution_tag>(cache)
            .variables(inertial_coordinates, tmpl::list<Poisson::Field>{}));

    // Compute error between numeric and analytic solutions
    const DataVector field_error = get(field) - get(field_analytic);

    // Compute l2 error squared over local element
    const double local_l2_error_square = alg::accumulate(
        field_error, 0.0, funcl::Plus<funcl::Identity, funcl::Square<>>{});

    // Collect volume data
    // Remove tensor types, only storing individual components
    std::vector<TensorComponent> components;
    components.reserve(3 + Dim);
    components.emplace_back(element_name + Poisson::Field::name(), get(field));
    components.emplace_back(element_name + Poisson::Field::name() + "Analytic",
                            get(field_analytic));
    components.emplace_back(element_name + Poisson::Field::name() + "Error",
                            field_error);
    components.emplace_back(element_name + "InertialCoordinates_x",
                            get<0>(inertial_coordinates));
    if (Dim >= 2) {
      components.emplace_back(element_name + "InertialCoordinates_y",
                              inertial_coordinates.get(1));
    }
    if (Dim >= 3) {
      components.emplace_back(element_name + "InertialCoordinates_z",
                              inertial_coordinates.get(2));
    }

    // Send data to volume observer
    auto& local_observer =
        *Parallel::get_parallel_component<observers::Observer<Metavariables>>(
             cache)
             .ckLocalBranch();
    Parallel::simple_action<observers::Actions::ContributeVolumeData>(
        local_observer,
        observers::ObservationId(
            iteration_id, typename Metavariables::element_observation_type{}),
        std::string{"/element_data"},
        observers::ArrayComponentId(
            std::add_pointer_t<ParallelComponent>{nullptr},
            Parallel::ArrayIndex<ElementIndex<Dim>>(array_index)),
        std::move(components), mesh.extents());

    // Send data to reduction observer
    Parallel::simple_action<observers::Actions::ContributeReductionData>(
        local_observer,
        observers::ObservationId(
            iteration_id, typename Metavariables::element_observation_type{}),
        std::string{"/element_data"},
        std::vector<std::string>{"Iteration", "NumberOfPoints", "L2Error"},
        observed_reduction_data{iteration_id, mesh.number_of_grid_points(),
                                local_l2_error_square});

    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace Actions
}  // namespace Poisson
