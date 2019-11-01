// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <tuple>
#include <utility>  // IWYU pragma: keep
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Variables.hpp"  // IWYU pragma: keep
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Initialization/Tags.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/Tags.hpp"
#include "Evolution/Systems/RadiationTransport/Tags.hpp"
#include "Evolution/TypeTraits.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace RadiationTransport {
namespace M1Grey {
namespace Actions {

struct InitializeM1Tags {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/, ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using system = typename Metavariables::system;
    using evolved_variables_tag = typename system::variables_tag;
    using hydro_variables_tag = typename system::hydro_variables_tag;
    using m1_variables_tag = typename system::primitive_variables_tag;
    // List of variables to be created... does NOT include
    // evolved_variables_tag because the evolved variables
    // are created by the ConservativeSystem initialization.
    using simple_tags =
        db::AddSimpleTags<hydro_variables_tag, m1_variables_tag>;
    using compute_tags = db::AddComputeTags<>;

    using EvolvedVars = typename evolved_variables_tag::type;
    using HydroVars = typename hydro_variables_tag::type;
    using M1Vars = typename m1_variables_tag::type;

    static constexpr size_t dim = system::volume_dim;
    const double initial_time = db::get<Initialization::Tags::InitialTime>(box);
    const size_t num_grid_points =
        db::get<::Tags::Mesh<dim>>(box).number_of_grid_points();
    const auto& inertial_coords =
        db::get<::Tags::Coordinates<dim, Frame::Inertial>>(box);

    db::mutate<evolved_variables_tag>(
        make_not_null(&box),
        make_overloader(
            [ initial_time, &inertial_coords ](
                const gsl::not_null<EvolvedVars*> evolved_vars,
                std::true_type /*is_analytic_solution*/,
                const auto& local_cache) noexcept {
              using solution_tag = ::Tags::AnalyticSolutionBase;
              evolved_vars->assign_subset(
                  Parallel::get<solution_tag>(local_cache)
                      .variables(inertial_coords, initial_time,
                                 typename evolved_variables_tag::tags_list{}));
            },
            [&inertial_coords](const gsl::not_null<EvolvedVars*> evolved_vars,
                               std::false_type /*is_analytic_solution*/,
                               const auto& local_cache) noexcept {
              using analytic_data_tag = ::Tags::AnalyticDataBase;
              evolved_vars->assign_subset(
                  Parallel::get<analytic_data_tag>(local_cache)
                      .variables(inertial_coords,
                                 typename evolved_variables_tag::tags_list{}));
            }),
        evolution::is_analytic_solution<typename Metavariables::initial_data>{},
        cache);

    // Get hydro variables
    HydroVars hydro_variables{num_grid_points};
    make_overloader(
        [ initial_time, &inertial_coords ](
            std::true_type /*is_analytic_solution*/,
            const gsl::not_null<HydroVars*> hydro_vars,
            const auto& local_cache) noexcept {
          using solution_tag = ::Tags::AnalyticSolutionBase;
          hydro_vars->assign_subset(
              Parallel::get<solution_tag>(local_cache)
                  .variables(inertial_coords, initial_time,
                             typename hydro_variables_tag::tags_list{}));
        },
        [&inertial_coords](std::false_type /*is_analytic_solution*/,
                           const gsl::not_null<HydroVars*> hydro_vars,
                           const auto& local_cache) noexcept {
          using analytic_data_tag = ::Tags::AnalyticDataBase;
          hydro_vars->assign_subset(
              Parallel::get<analytic_data_tag>(local_cache)
                  .variables(inertial_coords,
                             typename hydro_variables_tag::tags_list{}));
        })(
        evolution::is_analytic_solution<typename Metavariables::initial_data>{},
        make_not_null(&hydro_variables), cache);

    M1Vars m1_variables{num_grid_points, -1.};

    return std::make_tuple(
        db::create_from<db::RemoveTags<>, simple_tags, compute_tags>(
            std::move(box), std::move(hydro_variables),
            std::move(m1_variables)));
  }
};

}  // namespace Actions
}  // namespace M1Grey
}  // namespace RadiationTransport
