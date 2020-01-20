// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/Tags.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Evolution/TypeTraits.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits.hpp"

/// \cond
namespace Tags {
template <typename Tag>
struct Magnitude;
}  // namespace Tags
// IWYU pragma: no_forward_declare db::DataBox
/// \endcond

namespace dg {
namespace Actions {
/// \ingroup ActionsGroup
/// \ingroup DiscontinuousGalerkinGroup
/// \brief Packages data on external boundaries for calculating numerical flux.
/// Computes contributions on the interior side from the volume, and imposes
/// Dirichlet boundary conditions on the exterior side.
///
/// With:
/// - External<Tag> =
///   Tags::Interface<Tags::ExternalBoundaryDirections<volume_dim>, Tag>
///
/// Uses:
/// - ConstGlobalCache:
///   - Metavariables::boundary_condition_tag
/// - DataBox:
///   - Tags::Time
///   - External<Tags::BoundaryCoordinates<volume_dim>>,
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies:
///      - External<typename system::variables_tag>
///
/// \see ReceiveDataForFluxes
template <typename Metavariables>
struct ImposeDirichletBoundaryConditions {
 private:
  // BoundaryConditionMethod and BcSelector are used to select exactly how to
  // apply the Dirichlet boundary condition depending on properties of the
  // system. An overloaded `apply_impl` method is used that implements the
  // boundary condition calculation for the different types.
  enum class BoundaryConditionMethod {
    AnalyticBcNoPrimitives,
    AnalyticBcFluxConservativeWithPrimitives,
    Unknown
  };
  template <BoundaryConditionMethod Method>
  using BcSelector = std::integral_constant<BoundaryConditionMethod, Method>;

 public:
  using const_global_cache_tags =
      tmpl::list<typename Metavariables::boundary_condition_tag>;

  template <typename DbTags, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::ConstGlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    using system = typename Metavariables::system;
    return apply_impl<Metavariables::system::volume_dim>(
        box, cache,
        BcSelector<(not system::has_primitive_and_conservative_vars
                        ? BoundaryConditionMethod::AnalyticBcNoPrimitives
                        : (system::has_primitive_and_conservative_vars and
                                   system::is_in_flux_conservative_form
                               ? BoundaryConditionMethod::
                                     AnalyticBcFluxConservativeWithPrimitives
                               : BoundaryConditionMethod::Unknown))>{});
  }

 private:
  template <size_t VolumeDim, typename DbTags>
  static std::tuple<db::DataBox<DbTags>&&> apply_impl(
      db::DataBox<DbTags>& box,
      const Parallel::ConstGlobalCache<Metavariables>& cache,
      std::integral_constant<
          BoundaryConditionMethod,
          BoundaryConditionMethod::AnalyticBcNoPrimitives> /*meta*/) noexcept {
    using system = typename Metavariables::system;

    static_assert(
        system::is_in_flux_conservative_form or
            evolution::is_analytic_solution_v<
                typename Metavariables::boundary_condition_tag::type>,
        "Only analytic boundary conditions, or dirichlet boundary conditions "
        "for conservative systems are implemented");

    // Apply the boundary condition
    db::mutate_apply<tmpl::list<domain::Tags::Interface<
                         domain::Tags::BoundaryDirectionsExterior<VolumeDim>,
                         typename system::variables_tag>>,
                     tmpl::list<>>(
        [](const gsl::not_null<db::item_type<domain::Tags::Interface<
               domain::Tags::BoundaryDirectionsExterior<VolumeDim>,
               typename system::variables_tag>>*>
               external_bdry_vars,
           const double time, const auto& boundary_condition,
           const auto& boundary_coords) noexcept {
          for (auto& external_direction_and_vars : *external_bdry_vars) {
            auto& direction = external_direction_and_vars.first;
            auto& vars = external_direction_and_vars.second;
            vars.assign_subset(boundary_condition.variables(
                boundary_coords.at(direction), time,
                typename system::variables_tag::type::tags_list{}));
          }
        },
        make_not_null(&box), db::get<Tags::Time>(box),
        get<typename Metavariables::boundary_condition_tag>(cache),
        db::get<domain::Tags::Interface<
            domain::Tags::BoundaryDirectionsExterior<VolumeDim>,
            domain::Tags::Coordinates<VolumeDim, Frame::Inertial>>>(box));

    return std::forward_as_tuple(std::move(box));
  }

  template <typename TagsList, typename... ReturnTags, typename... ArgumentTags,
            typename System>
  static void apply_impl_helper_conservative_from_primitive(
      const gsl::not_null<Variables<TagsList>*> conservative_vars,
      const tuples::TaggedTuple<ArgumentTags...>& argument_vars,
      tmpl::list<ReturnTags...> /*meta*/, tmpl::list<System> /*meta*/
      ) noexcept {
    System::conservative_from_primitive::apply(
        make_not_null(&get<ReturnTags>(*conservative_vars))...,
        get<ArgumentTags>(argument_vars)...);
  }

  template <size_t VolumeDim, typename DbTags>
  static std::tuple<db::DataBox<DbTags>&&> apply_impl(
      db::DataBox<DbTags>& box,
      const Parallel::ConstGlobalCache<Metavariables>& cache,
      std::integral_constant<
          BoundaryConditionMethod,
          BoundaryConditionMethod::
              AnalyticBcFluxConservativeWithPrimitives> /*meta*/) noexcept {
    using system = typename Metavariables::system;

    static_assert(
        system::is_in_flux_conservative_form and
            system::has_primitive_and_conservative_vars and
            evolution::is_analytic_solution_v<
                typename Metavariables::boundary_condition_tag::type>,
        "Only analytic boundary conditions, or dirichlet boundary conditions "
        "for conservative systems are implemented");

    // Apply the boundary condition
    db::mutate_apply<tmpl::list<domain::Tags::Interface<
                         domain::Tags::BoundaryDirectionsExterior<VolumeDim>,
                         typename system::variables_tag>>,
                     tmpl::list<>>(
        [](const gsl::not_null<db::item_type<domain::Tags::Interface<
               domain::Tags::BoundaryDirectionsExterior<VolumeDim>,
               typename system::variables_tag>>*>
               external_bdry_vars,
           const double time, const auto& boundary_condition,
           const auto& boundary_coords) noexcept {
          for (auto& external_direction_and_vars : *external_bdry_vars) {
            auto& direction = external_direction_and_vars.first;
            auto& vars = external_direction_and_vars.second;

            apply_impl_helper_conservative_from_primitive(
                make_not_null(&vars),
                boundary_condition.variables(
                    boundary_coords.at(direction), time,
                    typename system::conservative_from_primitive::
                        argument_tags{}),
                typename system::conservative_from_primitive::return_tags{},
                tmpl::list<system>{});
          }
        },
        make_not_null(&box), db::get<Tags::Time>(box),
        get<typename Metavariables::boundary_condition_tag>(cache),
        db::get<domain::Tags::Interface<
            domain::Tags::BoundaryDirectionsExterior<VolumeDim>,
            domain::Tags::Coordinates<VolumeDim, Frame::Inertial>>>(box));

    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace Actions
}  // namespace dg
