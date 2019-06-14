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
#include "Evolution/Initialization/ConservativeSystem.hpp"
#include "Evolution/Initialization/DiscontinuousGalerkin.hpp"
#include "Evolution/Initialization/Domain.hpp"
#include "Evolution/Initialization/Evolution.hpp"
#include "Evolution/Initialization/Interface.hpp"
#include "Evolution/Initialization/Limiter.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/Tags.hpp"
#include "Evolution/Systems/RadiationTransport/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "Parallel/AddOptionsToDataBox.hpp"
#include "Parallel/ConstGlobalCache.hpp"
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
namespace detail {
template <class T, class = cpp17::void_t<>>
struct has_analytic_solution_alias : std::false_type {};
template <class T>
struct has_analytic_solution_alias<T,
                                   cpp17::void_t<typename T::analytic_solution>>
    : std::true_type {};
}  // namespace detail

template <size_t Dim>
struct Initialize {
  struct InitialExtents : db::SimpleTag {
    static std::string name() noexcept { return "InitialExtents"; }
    using type = std::vector<std::array<size_t, Dim>>;
  };
  struct Domain : db::SimpleTag {
    static std::string name() noexcept { return "Domain"; }
    using type = ::Domain<Dim, Frame::Inertial>;
  };
  struct InitialTime : db::SimpleTag {
    static std::string name() noexcept { return "InitialTime"; }
    using type = double;
  };
  struct InitialTimeDelta : db::SimpleTag {
    static std::string name() noexcept { return "InitialTimeDelta"; }
    using type = double;
  };
  struct InitialSlabSize : db::SimpleTag {
    static std::string name() noexcept { return "InitialSlabSize"; }
    using type = double;
  };

  using AddOptionsToDataBox = Parallel::ForwardAllOptionsToDataBox<tmpl::list<
      InitialExtents, Domain, InitialTime, InitialTimeDelta, InitialSlabSize>>;

  template <typename Metavariables>
  struct M1Tags {
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

    template <typename TagsList>
    static auto initialize(
        db::DataBox<TagsList>&& box,
        const Parallel::ConstGlobalCache<Metavariables>& cache,
        const double initial_time) noexcept {
      using EvolvedVars = typename evolved_variables_tag::type;
      using HydroVars = typename hydro_variables_tag::type;
      using M1Vars = typename m1_variables_tag::type;

      const size_t num_grid_points =
          db::get<::Tags::Mesh<Dim>>(box).number_of_grid_points();

      const auto& inertial_coords =
          db::get<::Tags::Coordinates<Dim, Frame::Inertial>>(box);

      // Get evolved M1 variables
      db::mutate<evolved_variables_tag>(
          make_not_null(&box),
          make_overloader(
              [ initial_time, &inertial_coords ](
                  const gsl::not_null<EvolvedVars*> evolved_vars,
                  std::true_type /*is_analytic_solution*/,
                  const auto& local_cache) noexcept {
                using solution_tag = OptionTags::AnalyticSolutionBase;
                evolved_vars->assign_subset(
                    Parallel::get<solution_tag>(local_cache)
                        .variables(
                            inertial_coords, initial_time,
                            typename evolved_variables_tag::tags_list{}));
              },
              [&inertial_coords](const gsl::not_null<EvolvedVars*> evolved_vars,
                                 std::false_type /*is_analytic_solution*/,
                                 const auto& local_cache) noexcept {
                using analytic_data_tag = OptionTags::AnalyticDataBase;
                evolved_vars->assign_subset(
                    Parallel::get<analytic_data_tag>(local_cache)
                        .variables(
                            inertial_coords,
                            typename evolved_variables_tag::tags_list{}));
              }),
          detail::has_analytic_solution_alias<Metavariables>{}, cache);

      // Get hydro variables
      HydroVars hydro_variables{num_grid_points};
      make_overloader(
          [ initial_time, &inertial_coords ](
              std::true_type /*is_analytic_solution*/,
              const gsl::not_null<HydroVars*> hydro_vars,
              const auto& local_cache) noexcept {
            using solution_tag = OptionTags::AnalyticSolutionBase;
            hydro_vars->assign_subset(
                Parallel::get<solution_tag>(local_cache)
                    .variables(inertial_coords, initial_time,
                               typename hydro_variables_tag::tags_list{}));
          },
          [&inertial_coords](std::false_type /*is_analytic_solution*/,
                             const gsl::not_null<HydroVars*> hydro_vars,
                             const auto& local_cache) noexcept {
            using analytic_data_tag = OptionTags::AnalyticDataBase;
            hydro_vars->assign_subset(
                Parallel::get<analytic_data_tag>(local_cache)
                    .variables(inertial_coords,
                               typename hydro_variables_tag::tags_list{}));
          })(detail::has_analytic_solution_alias<Metavariables>{},
             make_not_null(&hydro_variables), cache);

      M1Vars m1_variables{num_grid_points, -1.};

      return db::create_from<db::RemoveTags<>, simple_tags, compute_tags>(
          std::move(box), std::move(hydro_variables), std::move(m1_variables));
    }
  };

  template <typename System>
  struct GrTags {
    using gr_tag = typename System::spacetime_variables_tag;
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

      // Set initial data from analytic solution
      GrVars gr_vars{num_grid_points};
      make_overloader(
          [ initial_time, &inertial_coords ](
              std::true_type /*is_analytic_solution*/,
              const gsl::not_null<GrVars*> local_gr_vars,
              const auto& local_cache) noexcept {
            using solution_tag = OptionTags::AnalyticSolutionBase;
            local_gr_vars->assign_subset(
                Parallel::get<solution_tag>(local_cache)
                    .variables(inertial_coords, initial_time,
                               typename GrVars::tags_list{}));
          },
          [&inertial_coords](std::false_type /*is_analytic_solution*/,
                             const gsl::not_null<GrVars*> local_gr_vars,
                             const auto& local_cache) noexcept {
            using analytic_data_tag = OptionTags::AnalyticDataBase;
            local_gr_vars->assign_subset(
                Parallel::get<analytic_data_tag>(local_cache)
                    .variables(inertial_coords, typename GrVars::tags_list{}));
          })(detail::has_analytic_solution_alias<Metavariables>{},
             make_not_null(&gr_vars), cache);

      return db::create_from<db::RemoveTags<>, simple_tags, compute_tags>(
          std::move(box), std::move(gr_vars));
    }
  };

  template <class Metavariables>
  using return_tag_list = tmpl::append<
      typename Initialization::Domain<Dim>::simple_tags,
      typename GrTags<typename Metavariables::system>::simple_tags,
      typename Initialization::ConservativeSystem<
          typename Metavariables::system>::simple_tags,
      typename M1Tags<Metavariables>::simple_tags,
      typename Initialization::Interface<
          typename Metavariables::system>::simple_tags,
      typename Initialization::Evolution<
          typename Metavariables::system>::simple_tags,
      typename Initialization::DiscontinuousGalerkin<
          Metavariables>::simple_tags,
      typename Initialization::MinMod<Dim>::simple_tags,
      typename Initialization::Domain<Dim>::compute_tags,
      typename GrTags<typename Metavariables::system>::compute_tags,
      typename Initialization::ConservativeSystem<
          typename Metavariables::system>::compute_tags,
      typename M1Tags<Metavariables>::compute_tags,
      typename Initialization::Interface<
          typename Metavariables::system>::compute_tags,
      typename Initialization::Evolution<
          typename Metavariables::system>::compute_tags,
      typename Initialization::DiscontinuousGalerkin<
          Metavariables>::compute_tags,
      typename Initialization::MinMod<Dim>::compute_tags>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent,
            Requires<tmpl::list_contains_v<DbTagsList, Domain>> = nullptr>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ElementIndex<Dim>& array_index,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    const auto initial_extents = db::get<InitialExtents>(box);
    const auto initial_time = db::get<InitialTime>(box);
    const auto initial_dt = db::get<InitialTimeDelta>(box);
    const auto initial_slab_size = db::get<InitialSlabSize>(box);
    ::Domain<Dim, Frame::Inertial> domain{};
    db::mutate<Domain>(
        make_not_null(&box), [&domain](const auto domain_ptr) noexcept {
          domain = std::move(*domain_ptr);
        });
    auto initial_box =
        db::create_from<typename AddOptionsToDataBox::simple_tags>(
            std::move(box));

    using system = typename Metavariables::system;
    auto domain_box = Initialization::Domain<Dim>::initialize(
        db::DataBox<tmpl::list<>>{}, array_index, initial_extents, domain);
    auto gr_box =
        GrTags<system>::initialize(std::move(domain_box), cache, initial_time);
    auto system_box = Initialization::ConservativeSystem<system>::initialize(
        std::move(gr_box));
    auto m1_box = M1Tags<Metavariables>::initialize(std::move(system_box),
                                                    cache, initial_time);
    auto domain_interface_box =
        Initialization::Interface<system>::initialize(std::move(m1_box));
    auto evolution_box = Initialization::Evolution<system>::initialize(
        std::move(domain_interface_box), cache, initial_time, initial_dt,
        initial_slab_size);
    auto dg_box =
        Initialization::DiscontinuousGalerkin<Metavariables>::initialize(
            std::move(evolution_box), initial_extents);
    auto limiter_box =
        Initialization::MinMod<Dim>::initialize(std::move(dg_box));
    return std::make_tuple(std::move(limiter_box));
  }

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent,
            Requires<not tmpl::list_contains_v<DbTagsList, Domain>> = nullptr>
  static std::tuple<db::DataBox<DbTagsList>&&, bool> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ElementIndex<Dim>& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    return {std::move(box), true};
  }
};
}  // namespace Actions
}  // namespace M1Grey
}  // namespace RadiationTransport
