// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <type_traits>

#include "DataStructures/DataBox/Tag.hpp"
#include "Utilities/NoSuchType.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/CreateGetStaticMemberVariableOrDefault.hpp"
#include "Utilities/TypeTraits/CreateGetTypeAliasOrDefault.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

/// \cond
namespace Tags {
template <size_t Dim, typename TagsList>
struct BoundaryVariables;
}  // namespace Tags
/// \endcond

namespace evolution::dg {
namespace Tags {
/// \ingroup DiscontinuousGalerkinGroup
/// \brief The boundary-evolved twin of an interior source field.
///
/// A boundary-evolved field is stored and time-integrated only on external
/// boundary faces (a pointwise per-face-node ODE); it has no volume extent.
/// This is a prefix tag wrapping the interior source, so its type matches the
/// source.
///
/// \note Boundary evolved fields cannot currently be used with AMR or LTS.
template <typename Source>
struct BoundaryValue : db::PrefixTag, db::SimpleTag {
  using type = typename Source::type;
  /// The interior source field this boundary field is the twin of.
  using tag = Source;
};
}  // namespace Tags

namespace detail {
template <typename Tag>
struct is_boundary_variables_tag : std::false_type {};
template <size_t Dim, typename TagsList>
struct is_boundary_variables_tag<::Tags::BoundaryVariables<Dim, TagsList>>
    : std::true_type {};

template <typename VariablesTag, bool = tt::is_a_v<tmpl::list, VariablesTag>>
struct boundary_variables_tag_impl : std::false_type {
  using type = NoSuchType;
};
template <typename VariablesTag>
struct boundary_variables_tag_impl<VariablesTag, true> {
 private:
  using back = tmpl::back<VariablesTag>;

 public:
  static constexpr bool value = is_boundary_variables_tag<back>::value;
  static_assert(
      tmpl::count_if<VariablesTag,
                     tmpl::bind<is_boundary_variables_tag, tmpl::_1>>::value ==
          (value ? 1 : 0),
      "A list-valued variables_tag must contain at most one "
      "::Tags::BoundaryVariables entry, and it must be the last entry.");
  using type = tmpl::conditional_t<value, back, NoSuchType>;
};
}  // namespace detail

/// \ingroup DiscontinuousGalerkinGroup
/// \brief Whether the system declares boundary-evolved variables, i.e.
/// whether `System::variables_tag` is a `tmpl::list` whose last entry is a
/// `::Tags::BoundaryVariables`.
template <typename System>
constexpr bool system_has_boundary_variables_v =
    detail::boundary_variables_tag_impl<typename System::variables_tag>::value;

/// \ingroup DiscontinuousGalerkinGroup
/// \brief The `::Tags::BoundaryVariables` entry of the system's list-valued
/// `variables_tag`, or `NoSuchType` if the system does not declare
/// boundary-evolved variables.
template <typename System>
using boundary_variables_tag = typename detail::boundary_variables_tag_impl<
    typename System::variables_tag>::type;

/// @{
/// \ingroup DiscontinuousGalerkinGroup
/// \brief The projected interior inputs to a boundary condition's
/// `boundary_field_time_derivatives` method, in the order evolved,
/// primitive, temporary.
///
/// They are declared separately from the `dg_ghost` interior inputs
/// (`dg_interior_evolved_variables_tags` etc.) so that a boundary condition
/// can request a different projected interior state for its boundary-field
/// time derivative.  The boundary-condition pass projects the requested
/// fields onto the face exactly as it does for the `dg_ghost` inputs.
CREATE_GET_TYPE_ALIAS_OR_DEFAULT(
    boundary_field_time_derivatives_evolved_variables_tags)
CREATE_GET_TYPE_ALIAS_OR_DEFAULT(boundary_field_time_derivatives_primitive_tags)
CREATE_GET_TYPE_ALIAS_OR_DEFAULT(boundary_field_time_derivatives_temporary_tags)

template <typename BoundaryCondition>
using boundary_field_time_derivatives_interior_tags = tmpl::append<
    get_boundary_field_time_derivatives_evolved_variables_tags_or_default_t<
        BoundaryCondition, tmpl::list<>>,
    get_boundary_field_time_derivatives_primitive_tags_or_default_t<
        BoundaryCondition, tmpl::list<>>,
    get_boundary_field_time_derivatives_temporary_tags_or_default_t<
        BoundaryCondition, tmpl::list<>>>;
/// @}

namespace detail {
CREATE_GET_STATIC_MEMBER_VARIABLE_OR_DEFAULT(evolves_boundary_variables)

// Detect if the boundary condition declares a `boundary_field_time_derivatives`
// method.
template <typename BoundaryCondition, typename = std::void_t<>>
struct has_boundary_field_time_derivatives : std::false_type {};
template <typename BoundaryCondition>
struct has_boundary_field_time_derivatives<
    BoundaryCondition,
    std::void_t<decltype(&BoundaryCondition::boundary_field_time_derivatives)>>
    : std::true_type {};
template <typename BoundaryCondition>
constexpr bool has_boundary_field_time_derivatives_v =
    has_boundary_field_time_derivatives<BoundaryCondition>::value;
}  // namespace detail

/// @{
/// \ingroup DiscontinuousGalerkinGroup
/// \brief Whether a boundary condition evolves the system's boundary-evolved
/// variables: the opt-in is the marker
/// `static constexpr bool evolves_boundary_variables = true;` on the
/// boundary condition. An opt-in boundary condition must declare
/// the method `boundary_field_time_derivatives`.
template <typename BoundaryCondition>
constexpr bool evolves_boundary_variables_v =
    detail::get_evolves_boundary_variables_or_default_v<BoundaryCondition,
                                                        false>;
/// @}
}  // namespace evolution::dg
