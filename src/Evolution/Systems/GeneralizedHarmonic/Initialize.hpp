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
#include "Evolution/Initialization/DiscontinuousGalerkin.hpp"
#include "Evolution/Initialization/Domain.hpp"
#include "Evolution/Initialization/Evolution.hpp"
#include "Evolution/Initialization/Interface.hpp"
//#include "Evolution/Initialization/Limiter.hpp"
#include "Evolution/Initialization/NonConservativeSystem.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
//#include "Evolution/VariableFixing/FixToAtmosphere.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace GeneralizedHarmonic {
namespace Actions {
namespace detail {
// Note: this verifies if the Metavariables struct has an `analytic_solution`
// alias which we could use to initialize variables here.
template <class T, class = cpp17::void_t<>>
struct has_analytic_solution_alias : std::false_type {};
template <class T>
struct has_analytic_solution_alias<T,
                                   cpp17::void_t<typename T::analytic_solution>>
    : std::true_type {};
}  // namespace detail

template <size_t Dim>
struct Initialize {
  template <typename System>
  struct GrTags {
    using gr_tag = typename System::variables_tags;
    using simple_tags = db::AddSimpleTags<gr_tag>;
    using compute_tags = db::AddComputeTags<>;

    template <typename TagsList, typename Metavariables>
    static auto initialize(
        db::DataBox<TagsList>&& box,
        const Parallel::ConstGlobalCache<Metavariables>& cache,
        const double initial_time) noexcept {
      using GrVars = typename gr_tag::type;

      const size_t num_grid_points =
          db::get<::Tags::Mesh<Dim>>(box).number_of_grid_points();
      const auto& inertial_coords =
          db::get<::Tags::Coordinates<Dim, Frame::Inertial>>(box);

      // Set initial data from analytic solution or analytic data, whichever
      // is specified in the Metavariables struct.
      GrVars gr_vars{num_grid_points};
      make_overloader(
          [initial_time, &inertial_coords](
              std::true_type /*is_analytic_solution*/,
              const gsl::not_null<GrVars*> local_gr_vars,
              const auto& local_cache) noexcept {
            using analytic_solution_tag = OptionTags::AnalyticSolutionBase;
            local_gr_vars->assign_subset(
                Parallel::get<analytic_solution_tag>(local_cache)
                    .variables(inertial_coords, initial_time,
                               typename GrVars::tags_list{}));
          },
          [&inertial_coords](std::false_type /*is_analytic_solution*/,
                             const gsl::not_null<GrVars*> local_gr_vars,
                             const auto& local_cache) noexcept {
            using analytic_data_tag = OptionTags::AnalyticDataBase;
            local_gr_vars->assign_subset(
                Parallel::get<analytic_data_tag>(local_cache)
                    .variables(inertial_coords,
                               typename GrVars::tags_list{}));
          })(detail::has_analytic_solution_alias<Metavariables>{},
             make_not_null(&gr_vars), cache);

      return db::create_from<db::RemoveTags<>, simple_tags, compute_tags>(
          std::move(box), std::move(gr_vars));
    }
  };

  // Presumably, this is the list of tags to be initialized as part of
  // Initialization phase, and returned
  template <class Metavariables>
  using return_tag_list = tmpl::append<
      // Domain
      typename Initialization::Domain<Dim>::simple_tags,
      // Ev vars
      typename GrTags<typename Metavariables::system>::simple_tags,
      // NonConservative system
      typename Initialization::NonConservativeSystem<
          typename Metavariables::system>::simple_tags,
      // Interface
      typename Initialization::Interface<
          typename Metavariables::system>::simple_tags,
      // Evolution
      typename Initialization::Evolution<
          typename Metavariables::system>::simple_tags,
      // DG
      typename Initialization::DiscontinuousGalerkin<
          Metavariables>::simple_tags,
      // Domain - compute tags
      typename Initialization::Domain<Dim>::compute_tags,
      // Ev vars - compute tags
      typename GrTags<typename Metavariables::system>::compute_tags,
      // NonConservaive system - compute tags
      typename Initialization::NonConservativeSystem<
          typename Metavariables::system>::compute_tags,
      // Interface - compute tags
      typename Initialization::Interface<
          typename Metavariables::system>::compute_tags,
      // Evolution - compute tags
      typename Initialization::Evolution<
          typename Metavariables::system>::compute_tags,
      // DG - compute tags
      typename Initialization::DiscontinuousGalerkin<
          Metavariables>::compute_tags>;

  template <typename... InboxTags, typename Metavariables, typename ActionList,
            typename ParallelComponent>
  static auto apply(const db::DataBox<tmpl::list<>>& /* box */,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ElementIndex<Dim>& array_index,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    std::vector<std::array<size_t, Dim>> initial_extents,
                    Domain<Dim, Frame::Inertial> domain,
                    const double initial_time, const double initial_dt,
                    const double initial_slab_size) noexcept {
    using system = typename Metavariables::system;
    // Domain box
    auto domain_box = Initialization::Domain<Dim>::initialize(
        db::DataBox<tmpl::list<>>{}, array_index, initial_extents, domain);
    // Ev vars box
    auto gr_box =
        GrTags<system>::initialize(std::move(domain_box), cache, initial_time);
    // Evolution system box
    auto system_box = Initialization::NonConservativeSystem<system>::initialize(
        std::move(gr_box));
    // Interface box
    auto domain_interface_box =
        Initialization::Interface<system>::initialize(std::move(system_box));
    // Evolution box
    auto evolution_box = Initialization::Evolution<system>::initialize(
        std::move(domain_interface_box), cache, initial_time, initial_dt,
        initial_slab_size);
    // DG box
    auto dg_box =
        Initialization::DiscontinuousGalerkin<Metavariables>::initialize(
            std::move(evolution_box), initial_extents);

    // why return dg_box?
    return std::make_tuple(std::move(dg_box));
  }
};
}  // namespace Actions
}  // namespace GeneralizedHarmonic
