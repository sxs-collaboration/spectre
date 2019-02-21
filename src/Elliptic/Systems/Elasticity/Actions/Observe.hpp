// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Systems/Elasticity/Tags.hpp"
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

namespace Elasticity {
namespace Actions {

/*!
 * \brief Action to observe volume and reduction data for the Elasticity system
 *
 * This action observes the following:
 * - Reduction data:
 *   - `"Iteration"`: The linear solver iteration
 *   - `"NumberOfPoints"`: The total number of grid points across all elements
 *   - `"L2Error"`: The standard L2 vector norm of the pointwise difference
 * between the numerical and analytic solution \f$\sqrt{\sum_i
 * \frac{1}{D} \sum_j^\mathrm{D} \left( (u^j_\mathrm{numerical})_i -
 * (u^j_\mathrm{analytic})_i \right)^2}\f$ over the grid points across all
 * elements.
 * - Volume data:
 *   - `"Displacement": The numerical solution \f$u^i_\mathrm{numerical}\f$
 *   - `"DisplacementAnalytic"`: The analytic solution
 * \f$u^i_\mathrm{analytic}\f$
 *   - `"DisplacementError"`: The pointwise error \f$u^i_\mathrm{numerical} -
 * u^i_\mathrm{analytic}\f$
 *   - `"InertialCoordinates"`
 *
 * Uses:
 * - Metavariables:
 *   - `analytic_solution_tag`
 *   - Items required by `observers::Observer<Metavariables>`
 * - DataBox:
 *   - `LinearSolver::Tags::IterationId`
 *   - `Tags::Mesh<Dim>`
 *   - `Elasticity::Tags::Displacement`
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
  struct observation_type {};

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
    using field_tag = Elasticity::Tags::Displacement<Dim>;

    const auto& iteration_id = get<LinearSolver::Tags::IterationId>(box);
    const auto& mesh = get<::Tags::Mesh<Dim>>(box);
    const std::string element_name = MakeString{} << ElementId<Dim>(array_index)
                                                  << '/';

    // Retrieve the current numeric solution
    const auto& field = get<field_tag>(box);

    // Compute the analytic solution
    const auto& inertial_coordinates =
        db::get<::Tags::Coordinates<Dim, Frame::Inertial>>(box);
    const auto field_analytic = get<field_tag>(
        Parallel::get<typename Metavariables::analytic_solution_tag>(cache)
            .variables(inertial_coordinates, tmpl::list<field_tag>{}));

    // Collect volume and reduction data
    std::vector<TensorComponent> components;
    components.reserve(4 * Dim);
    auto field_error = make_with_value<db::item_type<field_tag>>(field, 0.);
    double local_l2_error_square = 0;
    for (size_t d = 0; d < Dim; d++) {
      // Compute error between numeric and analytic solutions
      field_error.get(d) = field.get(d) - field_analytic.get(d);
      // Compute l2 error squared over local element
      local_l2_error_square +=
          alg::accumulate(field_error.get(d), 0.0,
                          funcl::Plus<funcl::Identity, funcl::Square<>>{});
      const std::string component_suffix = d == 0 ? "_x" : d == 1 ? "_y" : "_z";
      components.emplace_back(
          element_name + field_tag::name() + component_suffix, field.get(d));
      components.emplace_back(
          element_name + field_tag::name() + "Analytic" + component_suffix,
          field_analytic.get(d));
      components.emplace_back(
          element_name + field_tag::name() + "Error" + component_suffix,
          field_error.get(d));
      components.emplace_back(
          element_name + "InertialCoordinates" + component_suffix,
          inertial_coordinates.get(d));
    }
    local_l2_error_square /= Dim;

    // Send data to volume observer
    auto& local_observer =
        *Parallel::get_parallel_component<observers::Observer<Metavariables>>(
             cache)
             .ckLocalBranch();
    Parallel::simple_action<observers::Actions::ContributeVolumeData>(
        local_observer,
        observers::ObservationId(iteration_id, observation_type{}),
        std::string{"/element_data"},
        observers::ArrayComponentId(
            std::add_pointer_t<ParallelComponent>{nullptr},
            Parallel::ArrayIndex<ElementIndex<Dim>>(array_index)),
        std::move(components), mesh.extents());

    // Send data to reduction observer
    Parallel::simple_action<observers::Actions::ContributeReductionData>(
        local_observer,
        observers::ObservationId(iteration_id, observation_type{}),
        std::string{"/element_data"},
        std::vector<std::string>{"Iteration", "NumberOfPoints", "L2Error"},
        observed_reduction_data{iteration_id.step_number,
                                mesh.number_of_grid_points(),
                                local_l2_error_square});

    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace Actions
}  // namespace Elasticity
