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
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
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
#include "Utilities/TypeTraits.hpp"

namespace grmhd {
namespace ValenciaDivClean {
namespace Actions {

struct InitializeGrTags {
  using initialization_tags = tmpl::list<Initialization::Tags::InitialTime>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/, ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using system = typename Metavariables::system;
    static constexpr size_t dim = system::volume_dim;
    using gr_tag = typename system::spacetime_variables_tag;
    using simple_tags = db::AddSimpleTags<gr_tag>;
    using compute_tags = db::AddComputeTags<>;

    const double initial_time = db::get<Initialization::Tags::InitialTime>(box);
    using GrVars = typename gr_tag::type;

    const size_t num_grid_points =
        db::get<::Tags::Mesh<dim>>(box).number_of_grid_points();
    const auto& inertial_coords =
        db::get<::Tags::Coordinates<dim, Frame::Inertial>>(box);

    // Set initial data from analytic solution
    GrVars gr_vars{num_grid_points};
    make_overloader(
        [ initial_time, &
          inertial_coords ](std::true_type /*is_analytic_solution*/,
                            const gsl::not_null<GrVars*> local_gr_vars,
                            const auto& local_cache) noexcept {
          using solution_tag = ::OptionTags::AnalyticSolutionBase;
          local_gr_vars->assign_subset(
              Parallel::get<solution_tag>(local_cache)
                  .variables(inertial_coords, initial_time,
                             typename GrVars::tags_list{}));
        },
        [&inertial_coords](std::false_type /*is_analytic_solution*/,
                           const gsl::not_null<GrVars*> local_gr_vars,
                           const auto& local_cache) noexcept {
          using analytic_data_tag = ::OptionTags::AnalyticDataBase;
          local_gr_vars->assign_subset(
              Parallel::get<analytic_data_tag>(local_cache)
                  .variables(inertial_coords, typename GrVars::tags_list{}));
        })(evolution::has_analytic_solution_alias<Metavariables>{},
           make_not_null(&gr_vars), cache);

    return std::make_tuple(
        Initialization::merge_into_databox<InitializeGrTags, simple_tags,
                                           compute_tags>(std::move(box),
                                                         std::move(gr_vars)));
  }
};
}  // namespace Actions
}  // namespace ValenciaDivClean
}  // namespace grmhd
