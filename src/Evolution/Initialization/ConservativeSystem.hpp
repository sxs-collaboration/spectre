// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Initialization/Tags.hpp"
#include "Evolution/TypeTraits.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits.hpp"

/// \cond
namespace Frame {
struct Inertial;
}  // namespace Frame
/// \endcond

namespace Initialization {
namespace Actions {
/// \ingroup InitializationGroup
/// \brief Allocate and set variables needed for evolution of conservative
/// systems
///
/// Uses:
/// - DataBox:
///   * `Tags::Mesh<Dim>`
///
/// DataBox changes:
/// - Adds:
///   * System::variables_tag
///   * db::add_tag_prefix<Tags::Flux, System::variables_tag>
///   * db::add_tag_prefix<Tags::Source, System::variables_tag>
///
/// - Removes: nothing
/// - Modifies: nothing
struct ConservativeSystem {
  using initialization_tags = tmpl::list<Initialization::Tags::InitialTime>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<tmpl::list_contains_v<
                typename db::DataBox<DbTagsList>::simple_item_tags,
                Initialization::Tags::InitialTime>> = nullptr>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/, ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using system = typename Metavariables::system;
    static_assert(system::is_in_flux_conservative_form,
                  "System is not in flux conservative form");
    static constexpr size_t dim = system::volume_dim;
    using variables_tag = typename system::variables_tag;
    using fluxes_tag = db::add_tag_prefix<::Tags::Flux, variables_tag,
                                          tmpl::size_t<dim>, Frame::Inertial>;
    using sources_tag = db::add_tag_prefix<::Tags::Source, variables_tag>;
    using simple_tags =
        db::AddSimpleTags<variables_tag, fluxes_tag, sources_tag>;
    using compute_tags = db::AddComputeTags<>;

    const size_t num_grid_points =
        db::get<::Tags::Mesh<dim>>(box).number_of_grid_points();
    typename variables_tag::type vars(num_grid_points);
    typename fluxes_tag::type fluxes(num_grid_points);
    typename sources_tag::type sources(num_grid_points);

    return std::make_tuple(initialize_vars(
        merge_into_databox<ConservativeSystem, simple_tags, compute_tags>(
            std::move(box), std::move(vars), std::move(fluxes),
            std::move(sources)),
        cache));
  }

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<not tmpl::list_contains_v<
                typename db::DataBox<DbTagsList>::simple_item_tags,
                Initialization::Tags::InitialTime>> = nullptr>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& /*box*/,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    ERROR(
        "Could not find dependency 'Initialization::Tags::InitialTime' in "
        "DataBox.");
  }

 private:
  template <
      typename DbTagsList, typename Metavariables,
      Requires<Metavariables::system::has_primitive_and_conservative_vars> =
          nullptr>
  static auto initialize_vars(
      db::DataBox<DbTagsList>&& box,
      const Parallel::ConstGlobalCache<Metavariables>& cache) noexcept {
    using system = typename Metavariables::system;
    static constexpr size_t dim = system::volume_dim;
    using primitives_tag = typename system::primitive_variables_tag;
    using simple_tags =
        db::AddSimpleTags<primitives_tag,
                          typename Metavariables::equation_of_state_tag>;
    using compute_tags = db::AddComputeTags<>;

    const double initial_time = db::get<Initialization::Tags::InitialTime>(box);
    using PrimitiveVars = typename primitives_tag::type;

    const size_t num_grid_points =
        db::get<::Tags::Mesh<dim>>(box).number_of_grid_points();

    const auto& inertial_coords =
        db::get<::Tags::Coordinates<dim, Frame::Inertial>>(box);

    // Set initial data from analytic solution
    PrimitiveVars primitive_vars{num_grid_points};
    auto equation_of_state = make_overloader(
        [ initial_time, &
          inertial_coords ](std::true_type /*is_analytic_solution*/,
                            const gsl::not_null<PrimitiveVars*> prim_vars,
                            const auto& local_cache) noexcept {
          using solution_tag = ::Tags::AnalyticSolutionBase;
          prim_vars->assign_subset(
              Parallel::get<solution_tag>(local_cache)
                  .variables(
                      inertial_coords, initial_time,
                      typename Metavariables::analytic_variables_tags{}));
          return Parallel::get<solution_tag>(local_cache).equation_of_state();
        },
        [&inertial_coords](std::false_type /*is_analytic_solution*/,
                           const gsl::not_null<PrimitiveVars*> prim_vars,
                           const auto& local_cache) noexcept {
          using analytic_data_tag = ::Tags::AnalyticDataBase;
          prim_vars->assign_subset(
              Parallel::get<analytic_data_tag>(local_cache)
                  .variables(
                      inertial_coords,
                      typename Metavariables::analytic_variables_tags{}));
          return Parallel::get<analytic_data_tag>(local_cache)
              .equation_of_state();
        })(
        evolution::is_analytic_solution<typename Metavariables::initial_data>{},
        make_not_null(&primitive_vars), cache);

    return Initialization::merge_into_databox<ConservativeSystem, simple_tags,
                                              compute_tags>(
        std::move(box), std::move(primitive_vars),
        std::move(equation_of_state));
  }

   template <typename DbTagsList, typename Metavariables,
             Requires<not Metavariables::system::
                          has_primitive_and_conservative_vars> = nullptr>
   static auto initialize_vars(
       db::DataBox<DbTagsList>&& box,
       const Parallel::ConstGlobalCache<Metavariables>& cache) noexcept {
     using system = typename Metavariables::system;
     static constexpr size_t dim = system::volume_dim;
     using variables_tag = typename system::variables_tag;

     const double initial_time =
         db::get<Initialization::Tags::InitialTime>(box);
     const auto& inertial_coords =
         db::get<::Tags::Coordinates<dim, Frame::Inertial>>(box);

     // Set initial data from analytic solution
     using Vars = typename variables_tag::type;
     using solution_tag = ::Tags::AnalyticSolutionBase;
     db::mutate<variables_tag>(
         make_not_null(&box), [&cache, &inertial_coords, initial_time ](
                                  const gsl::not_null<Vars*> vars) noexcept {
           vars->assign_subset(Parallel::get<solution_tag>(cache).variables(
               inertial_coords, initial_time, typename Vars::tags_list{}));
         });

     return std::move(box);
   }
};
}  // namespace Actions
}  // namespace Initialization
