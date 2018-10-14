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
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/LorentzFactor.hpp"
#include "PointwiseFunctions/Hydro/SpecificEnthalpy.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd {
namespace ValenciaDivClean {
namespace Actions {
// Note:  I've left the Dim and System template parameters until it is clear
// whether or not what remains is specific to this system, and what might be
// applicable to more than one system
template <size_t Dim>
struct Initialize {
  template <typename Metavariables>
  struct PrimitiveTags {
    using system = typename Metavariables::system;
    using primitives_tag = typename system::primitive_variables_tag;
    using simple_tags = db::AddSimpleTags<
        primitives_tag, typename Metavariables::equation_of_state_tag,
        grmhd::ValenciaDivClean::Tags::ConstraintDampingParameter>;
    using compute_tags = db::AddComputeTags<>;

    template <typename TagsList>
    static auto initialize(
        db::DataBox<TagsList>&& box,
        const Parallel::ConstGlobalCache<Metavariables>& cache,
        const double initial_time) noexcept {
      using PrimitiveVars = typename primitives_tag::type;

      const size_t num_grid_points =
          db::get<::Tags::Mesh<Dim>>(box).number_of_grid_points();

      const auto& inertial_coords =
          db::get<::Tags::Coordinates<Dim, Frame::Inertial>>(box);

      // Set initial data from analytic solution
      using solution_tag = OptionTags::AnalyticSolutionBase;
      PrimitiveVars primitive_vars{num_grid_points};
      primitive_vars.assign_subset(Parallel::get<solution_tag>(cache).variables(
          inertial_coords, initial_time,
          typename Metavariables::analytic_variables_tags{}));

      auto equation_of_state =
          Parallel::get<solution_tag>(cache).equation_of_state();

      return db::create_from<db::RemoveTags<>, simple_tags, compute_tags>(
          std::move(box), std::move(primitive_vars),
          std::move(equation_of_state),
          Parallel::get<OptionTags::DampingParameter>(cache));
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
      using solution_tag = OptionTags::AnalyticSolutionBase;
      GrVars gr_vars{num_grid_points};
      gr_vars.assign_subset(
          Parallel::get<solution_tag>(cache).background_spacetime().variables(
              inertial_coords, initial_time, typename GrVars::tags_list{}));

      return db::create_from<db::RemoveTags<>, simple_tags, compute_tags>(
          std::move(box), std::move(gr_vars));
    }
  };

  template <class Metavariables>
  using return_tag_list = tmpl::append<
      typename Initialization::Domain<Dim>::simple_tags,
      typename GrTags<typename Metavariables::system>::simple_tags,
      typename PrimitiveTags<Metavariables>::simple_tags,
      typename Initialization::ConservativeSystem<
          typename Metavariables::system>::simple_tags,
      typename Initialization::Interface<
          typename Metavariables::system>::simple_tags,
      typename Initialization::Evolution<
          typename Metavariables::system>::simple_tags,
      typename Initialization::DiscontinuousGalerkin<
          Metavariables>::simple_tags,
      typename Initialization::MinMod<Dim>::simple_tags,
      typename Initialization::Domain<Dim>::compute_tags,
      typename GrTags<typename Metavariables::system>::compute_tags,
      typename PrimitiveTags<Metavariables>::compute_tags,
      typename Initialization::ConservativeSystem<
          typename Metavariables::system>::compute_tags,
      typename Initialization::Interface<
          typename Metavariables::system>::compute_tags,
      typename Initialization::Evolution<
          typename Metavariables::system>::compute_tags,
      typename Initialization::DiscontinuousGalerkin<
          Metavariables>::compute_tags,
      typename Initialization::MinMod<Dim>::compute_tags>;

  template <typename... InboxTags, typename Metavariables, typename ActionList,
            typename ParallelComponent>
  static auto apply(const db::DataBox<tmpl::list<>>& /*box*/,
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
    auto domain_box = Initialization::Domain<Dim>::initialize(
        db::DataBox<tmpl::list<>>{}, array_index, initial_extents, domain);
    auto gr_box = GrTags<system>::initialize(std::move(domain_box), cache,
                                             initial_time);
    auto primitive_box = PrimitiveTags<Metavariables>::initialize(
        std::move(gr_box), cache, initial_time);
    auto system_box = Initialization::ConservativeSystem<system>::initialize(
        std::move(primitive_box));
    auto conservative_box =
        Initialization::ConservativeVars<system>::initialize(
            std::move(system_box));
    auto domain_interface_box = Initialization::Interface<system>::initialize(
        std::move(conservative_box));
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
};
}  // namespace Actions
}  // namespace ValenciaDivClean
}  // namespace grmhd
