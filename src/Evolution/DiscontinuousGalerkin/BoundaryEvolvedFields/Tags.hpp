// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <type_traits>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Time/History.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/CreateGetTypeAliasOrDefault.hpp"

namespace evolution::dg {
namespace Tags {
/// \ingroup DiscontinuousGalerkinGroup
/// \brief The boundary-evolved twin of an interior source field.
///
/// A boundary-evolved field is stored and time-integrated only on external
/// boundary faces (a pointwise per-face-node ODE); it
/// has no volume extent. This is a prefix tag wrapping the
/// interior source, so its type matches the source, the
/// initialization can be automatic, and its name disambiguates the
/// source (`db::tag_name` gives e.g. "BoundaryValue(Psi)").
///
/// \note Boundary evolved fields cannot currently be used with AMR.
template <typename Source>
struct BoundaryValue : db::PrefixTag, db::SimpleTag {
  using type = typename Source::type;
  /// The interior source field this boundary field is the twin of.
  using tag = Source;
};

/// \ingroup DiscontinuousGalerkinGroup
/// \brief The current values of the boundary-evolved fields, one contiguous
/// face-sized `Variables` per opting external face.
///
/// Keyed by external `Direction`; empty on interior elements and on non-opting
/// faces. Per-face storage is required because a corner/edge node shared by
/// several external faces carries a distinct boundary value per face (each
/// face applies its boundary condition with its own normal, so the values
/// differ even when two faces hold the same boundary condition).
template <size_t Dim, typename FieldTagsList>
struct BoundaryEvolvedFieldsValues : db::SimpleTag {
  using type = DirectionMap<Dim, Variables<FieldTagsList>>;
};

/// \ingroup DiscontinuousGalerkinGroup
/// \brief The per-face stash holding the time derivatives of the
/// boundary-evolved fields, produced by the `boundary_field_time_derivatives`
/// defined in a concrete boundary condition class and consumed when recording
/// the time-stepper history.
///
/// The stored type is the time stepper's `DerivVars`
/// (`db::prefix_variables<::Tags::dt, Variables<FieldTagsList>>`) so that it
/// passes the type checks in `TimeSteppers::History::insert`.
template <size_t Dim, typename FieldTagsList>
struct BoundaryEvolvedFieldsDtStash : db::SimpleTag {
  using type = DirectionMap<
      Dim, typename TimeSteppers::History<Variables<FieldTagsList>>::DerivVars>;
};

/// \ingroup DiscontinuousGalerkinGroup
/// \brief The per-face time-stepper history of the boundary-evolved fields.
///
/// A `DirectionMap` of one `TimeSteppers::History` per opting external face,
/// deliberately a plain `db::SimpleTag` rather than a
/// `Tags::HistoryEvolvedVariables`. The volume variables already own the
/// element's single history; a *second* `HistoryEvolvedVariables` would break
/// code that assumes exactly one history. A plain tag is invisible to
/// `Tags::get_all_history_tags` (which filters that type), so
/// the facility's update action instead syncs its order to the volume history.
template <size_t Dim, typename FieldTagsList>
struct BoundaryEvolvedFieldsHistory : db::SimpleTag {
  using type =
      DirectionMap<Dim, TimeSteppers::History<Variables<FieldTagsList>>>;
};
}  // namespace Tags

namespace BoundaryEvolvedFields {
namespace detail {
CREATE_GET_TYPE_ALIAS_OR_DEFAULT(boundary_evolved_variables)
}  // namespace detail

/// \ingroup DiscontinuousGalerkinGroup
/// \brief The boundary-evolved field tags declared by a boundary condition, or
/// an empty list if it does not opt in.
///
/// A boundary condition opts in by declaring
/// `using boundary_evolved_variables =
/// tmpl::list<Tags::BoundaryValue<Source>...>;`.
template <typename BoundaryCondition>
using boundary_evolved_variables_of =
    detail::get_boundary_evolved_variables_or_default_t<BoundaryCondition,
                                                        tmpl::list<>>;

/// \ingroup DiscontinuousGalerkinGroup
/// \brief `true` if the boundary condition opts into boundary-evolved fields.
template <typename BoundaryCondition>
constexpr bool bc_opts_in_v =
    not std::is_same_v<boundary_evolved_variables_of<BoundaryCondition>,
                       tmpl::list<>>;

namespace detail {
// Each boundary condition's declared `boundary_evolved_variables`, in list
// order; `tmpl::list<>` for non-opting conditions.
template <typename DerivedBoundaryConditionsList>
using declared_boundary_evolved_lists =
    tmpl::transform<DerivedBoundaryConditionsList,
                    tmpl::bind<boundary_evolved_variables_of, tmpl::_1>>;
}  // namespace detail

/// \ingroup DiscontinuousGalerkinGroup
/// \brief The flat, duplicate-free union of the boundary-evolved field tags
/// declared by all boundary conditions in `DerivedBoundaryConditionsList`.
///
/// This is the compile-time field set the facility stores and integrates. It is
/// empty when no boundary condition opts in. An empty union is effectively a
/// no-op for the boundary field facility.
template <typename DerivedBoundaryConditionsList>
using boundary_evolved_field_tags = tmpl::remove_duplicates<tmpl::flatten<
    detail::declared_boundary_evolved_lists<DerivedBoundaryConditionsList>>>;

/// \ingroup DiscontinuousGalerkinGroup
/// \brief `true` if every boundary condition in `DerivedBoundaryConditionsList`
/// that opts into the facility declares the identical
/// `boundary_evolved_variables` list (ordering in each list must be the same).
///
/// At most one distinct non-empty declared list exists across
/// the boundary conditions. Heterogeneous per-face field sets are not yet
/// supported.
template <typename DerivedBoundaryConditionsList>
constexpr bool boundary_evolved_fields_are_homogeneous_v =
    tmpl::size<tmpl::remove_duplicates<tmpl::remove<
        detail::declared_boundary_evolved_lists<DerivedBoundaryConditionsList>,
        tmpl::list<>>>>::value <= 1;
}  // namespace BoundaryEvolvedFields
}  // namespace evolution::dg
