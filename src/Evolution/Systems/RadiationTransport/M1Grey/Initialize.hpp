// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <utility>  // IWYU pragma: keep  // for move

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/Initialization/InitialData.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Frame {
struct Inertial;
}  // namespace Frame
namespace Initialization {
namespace Tags {
struct InitialTime;
}  // namespace Tags
}  // namespace Initialization
namespace Tags {
struct AnalyticSolutionOrData;
}  // namespace Tags
namespace domain {
namespace Tags {
template <size_t Dim, typename Frame>
struct Coordinates;
template <size_t VolumeDim>
struct Mesh;
}  // namespace Tags
}  // namespace domain
// IWYU pragma: no_forward_declare db::DataBox
/// \endcond

namespace RadiationTransport {
namespace M1Grey {
namespace Actions {

template <typename System>
struct InitializeM1Tags {
  using evolved_variables_tag = typename System::variables_tag;
  using hydro_variables_tag = typename System::hydro_variables_tag;
  using m1_variables_tag = typename System::primitive_variables_tag;
  // List of variables to be created... does NOT include
  // evolved_variables_tag because the evolved variables
  // are created by the ConservativeSystem initialization.
  using simple_tags = tmpl::list<hydro_variables_tag, m1_variables_tag>;
  using compute_tags = tmpl::list<>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/, ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) {
    using EvolvedVars = typename evolved_variables_tag::type;
    using HydroVars = typename hydro_variables_tag::type;
    using M1Vars = typename m1_variables_tag::type;

    static constexpr size_t dim = System::volume_dim;
    const double initial_time = db::get<Initialization::Tags::InitialTime>(box);
    const size_t num_grid_points =
        db::get<domain::Tags::Mesh<dim>>(box).number_of_grid_points();
    const auto& inertial_coords =
        db::get<domain::Tags::Coordinates<dim, Frame::Inertial>>(box);

    db::mutate<evolved_variables_tag>(
        make_not_null(&box),
        [&cache, initial_time,
         &inertial_coords](const gsl::not_null<EvolvedVars*> evolved_vars) {
          evolved_vars->assign_subset(evolution::Initialization::initial_data(
              Parallel::get<::Tags::AnalyticSolutionOrData>(cache),
              inertial_coords, initial_time,
              typename evolved_variables_tag::tags_list{}));
        });

    // Get hydro variables
    HydroVars hydro_variables{num_grid_points};
    hydro_variables.assign_subset(evolution::Initialization::initial_data(
        Parallel::get<::Tags::AnalyticSolutionOrData>(cache), inertial_coords,
        initial_time, typename hydro_variables_tag::tags_list{}));

    M1Vars m1_variables{num_grid_points, -1.};
    Initialization::mutate_assign<simple_tags>(make_not_null(&box),
                                               std::move(hydro_variables),
                                               std::move(m1_variables));

    return std::make_tuple(std::move(box));
  }
};

}  // namespace Actions
}  // namespace M1Grey
}  // namespace RadiationTransport
