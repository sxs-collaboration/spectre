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
//#include "DataStructures/Variables.hpp"  // IWYU pragma: keep
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Initialization/DiscontinuousGalerkin.hpp"
#include "Evolution/Initialization/Domain.hpp"
#include "Evolution/Initialization/Evolution.hpp"
#include "Evolution/Initialization/Interface.hpp"
#include "Evolution/Initialization/NonConservativeSystem.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeGhQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits.hpp"

namespace GeneralizedHarmonic {
namespace Actions {
namespace detail {
/*
 * Note: Check whether the Metavariables struct has an `analytic_solution`
 * alias or an `analytic_data` alias, either of which could be used to
 * initialize evolution variables.
 */
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
  struct GhTags {
    using gh_tags = typename System::variables_tag;
    using simple_tags = db::AddSimpleTags<gh_tags>;
    using compute_tags = db::AddComputeTags<>;

    template <typename TagsList, typename Metavariables>
    static auto initialize(
        db::DataBox<TagsList>&& box,
        const Parallel::ConstGlobalCache<Metavariables>& cache,
        const double initial_time) noexcept {
      using frame = Frame::Inertial;
      using GhVars = typename gh_tags::type;

      const size_t num_grid_points =
          db::get<::Tags::Mesh<Dim>>(box).number_of_grid_points();
      const auto& inertial_coords =
          db::get<::Tags::Coordinates<Dim, frame>>(box);

      // Set initial data from analytic solution
      GhVars gh_vars{num_grid_points};
      make_overloader([ initial_time, &inertial_coords ](
                          std::true_type /*is_analytic_solution*/,
                          const gsl::not_null<GhVars*> local_gh_vars,
                          const auto& local_cache) noexcept {
        using analytic_solution_tag = OptionTags::AnalyticSolutionBase;
        /*
         * It is assumed here that the analytic solution adds the following
         * foliation-related variables to the databox (only):
         * 1. Lapse, \partial_t Lapse, \partial_i Lapse
         * 2. Shift, \partial_t Shift, \partial_i Shift
         * 3. SpatialMetric, \partial_0 SpatialMetric,
         *    \partial_i SpatialMetric.
         */
        const auto& solution_vars =
            Parallel::get<analytic_solution_tag>(local_cache)
                .variables(inertial_coords, initial_time,
                           typename gr::Solutions::KerrSchild::template tags<
                               DataVector>{});
        // First fetch lapse, shift, spatial metric and their derivs
        const auto& lapse = get<gr::Tags::Lapse<DataVector>>(solution_vars);
        const auto& dt_lapse =
            get<::Tags::dt<gr::Tags::Lapse<DataVector>>>(solution_vars);
        const auto& deriv_lapse =
            get<::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<Dim>,
                              frame>>(solution_vars);

        const auto& shift =
            get<gr::Tags::Shift<Dim, frame, DataVector>>(solution_vars);
        const auto& dt_shift =
            get<::Tags::dt<gr::Tags::Shift<Dim, frame, DataVector>>>(
                solution_vars);
        const auto& deriv_shift =
            get<::Tags::deriv<gr::Tags::Shift<Dim, frame, DataVector>,
                              tmpl::size_t<Dim>, frame>>(solution_vars);

        const auto& spatial_metric =
            get<gr::Tags::SpatialMetric<Dim, frame, DataVector>>(solution_vars);
        const auto& dt_spatial_metric =
            get<::Tags::dt<gr::Tags::SpatialMetric<Dim, frame, DataVector>>>(
                solution_vars);
        const auto& deriv_spatial_metric =
            get<::Tags::deriv<gr::Tags::SpatialMetric<Dim, frame, DataVector>,
                              tmpl::size_t<Dim>, frame>>(solution_vars);

        const auto& spacetime_metric =
            ::gr::spacetime_metric<Dim, frame, DataVector>(lapse, shift,
                                                           spatial_metric);
        const auto& phi = GeneralizedHarmonic::phi<Dim, frame, DataVector>(
            lapse, deriv_lapse, shift, deriv_shift, spatial_metric,
            deriv_spatial_metric);
        const auto& pi = GeneralizedHarmonic::pi<Dim, frame, DataVector>(
            lapse, dt_lapse, shift, dt_shift, spatial_metric, dt_spatial_metric,
            phi);

        const tuples::TaggedTuple<gr::Tags::SpacetimeMetric<Dim>,
                                  GeneralizedHarmonic::Tags::Phi<Dim>,
                                  GeneralizedHarmonic::Tags::Pi<Dim>>
            gh_solution_tuple(spacetime_metric, phi, pi);

        local_gh_vars->assign_subset(gh_solution_tuple);
      },
                      [&inertial_coords](
                          std::false_type /*is_analytic_solution*/,
                          const gsl::not_null<GhVars*> local_gh_vars,
                          const auto& local_cache) noexcept {
                        using analytic_data_tag = OptionTags::AnalyticDataBase;
                        local_gh_vars->assign_subset(
                            Parallel::get<analytic_data_tag>(local_cache)
                                .variables(inertial_coords,
                                           typename GhVars::tags_list{}));
                      })(detail::has_analytic_solution_alias<Metavariables>{},
                         make_not_null(&gh_vars), cache);

      return db::create_from<db::RemoveTags<>, simple_tags, compute_tags>(
          std::move(box), std::move(gh_vars));
    }  // initialize
  };

  // Presumably, this is the list of tags to be initialized as part of
  // Initialization phase, and returned
  template <class Metavariables>
  using return_tag_list = tmpl::append<
      // Domain - simple tags
      typename Initialization::Domain<Dim>::simple_tags,
      // GH - simple tags
      typename GhTags<typename Metavariables::system>::simple_tags,
      // NonConservative system - simple tags
      typename Initialization::NonConservativeSystem<
          typename Metavariables::system>::simple_tags,
      // Interface - simple tags
      // typename Initialization::Interface<
      // typename Metavariables::system>::simple_tags,
      // Evolution - simple tags
      // typename Initialization::Evolution<
      // typename Metavariables::system>::simple_tags,
      // DG - simple tags
      // typename Initialization::DiscontinuousGalerkin<
      // Metavariables>::simple_tags,
      // Domain - compute tags
      typename Initialization::Domain<Dim>::compute_tags,
      // GH - simple tags
      typename GhTags<typename Metavariables::system>::compute_tags,
      // NonConservaive system - compute tags
      typename Initialization::NonConservativeSystem<
          typename Metavariables::system>::compute_tags
      // Interface - compute tags
      // typename Initialization::Interface<
      // typename Metavariables::system>::compute_tags,
      // Evolution - compute tags
      // typename Initialization::Evolution<
      // typename Metavariables::system>::compute_tags,
      // DG - compute tags
      // typename Initialization::DiscontinuousGalerkin<
      // Metavariables>::compute_tags
      >;

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
    // Generalized Harmonic box
    auto gh_box =
        GhTags<system>::initialize(std::move(domain_box), cache, initial_time);
    // Evolution system box
    auto system_box = Initialization::NonConservativeSystem<system>::initialize(
        std::move(gh_box));
    // Interface box
    // auto domain_interface_box =
    // Initialization::Interface<system>::initialize(std::move(system_box));
    // Evolution box
    // auto evolution_box = Initialization::Evolution<system>::initialize(
    // std::move(domain_interface_box), cache, initial_time, initial_dt,
    // initial_slab_size);
    // DG box
    // auto dg_box =
    // Initialization::DiscontinuousGalerkin<Metavariables>::initialize(
    // std::move(evolution_box), initial_extents);

    return std::make_tuple(std::move(system_box));
  }
};
}  // namespace Actions
}  // namespace GeneralizedHarmonic
