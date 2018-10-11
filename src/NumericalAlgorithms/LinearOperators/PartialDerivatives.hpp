// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines functions computing partial derivatives.

#pragma once

#include <array>
#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Variables.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

/// \cond
class DataVector;
template <size_t Dim>
class Mesh;

namespace Tags {
template <size_t Dim>
struct Mesh;
template <class TagList>
struct Variables;
/// \endcond

/*!
 * \ingroup DataBoxTagsGroup
 * \brief Prefix indicating spatial derivatives
 *
 * Prefix indicating the spatial derivatives of a Tensor or that a Variables
 * contains spatial derivatives of Tensors.
 *
 * \tparam Tag The tag to wrap
 * \tparam Dim The volume dim as a type (e.g. `tmpl::size_t<Dim>`)
 * \tparam Frame The frame of the derivative index
 *
 * \see Tags::DerivCompute
 */
template <typename Tag, typename Dim, typename Frame, typename = std::nullptr_t>
struct deriv;

template <typename Tag, typename Dim, typename Frame>
struct deriv<Tag, Dim, Frame, Requires<tt::is_a_v<Tensor, db::item_type<Tag>>>>
    : db::PrefixTag, db::SimpleTag {
  using type =
      TensorMetafunctions::prepend_spatial_index<db::item_type<Tag>, Dim::value,
                                                 UpLo::Lo, Frame>;
  using tag = Tag;
  static std::string name() noexcept { return "deriv(" + Tag::name() + ")"; }
};
template <typename Tag, typename Dim, typename Frame>
struct deriv<Tag, Dim, Frame,
             Requires<tt::is_a_v<::Variables, db::item_type<Tag>>>>
    : db::PrefixTag, db::SimpleTag {
  using type = db::item_type<Tag>;
  using tag = Tag;
  static std::string name() noexcept { return "deriv(" + Tag::name() + ")"; }
};

}  // namespace Tags

// @{
/// \ingroup NumericalAlgorithmsGroup
/// \brief Compute the partial derivatives of each variable with respect to
/// the logical coordinate.
///
/// \requires `DerivativeTags` to be the head of `VariableTags`
///
/// Returns a `Variables` with a spatial tensor index appended to the front
/// of each tensor within `u` and each `Tag` wrapped with a `Tags::deriv`.
///
/// \tparam DerivativeTags the subset of `VariableTags` for which derivatives
/// are computed.
template <typename DerivativeTags, typename VariableTags, size_t Dim>
void logical_partial_derivatives(
    gsl::not_null<std::array<Variables<DerivativeTags>, Dim>*>
        logical_partial_derivatives_of_u,
    const Variables<VariableTags>& u, const Mesh<Dim>& mesh) noexcept;

template <typename DerivativeTags, typename VariableTags, size_t Dim>
auto logical_partial_derivatives(const Variables<VariableTags>& u,
                                 const Mesh<Dim>& mesh) noexcept
    -> std::array<Variables<DerivativeTags>, Dim>;
// @}

// @{
/// \ingroup NumericalAlgorithmsGroup
/// \brief Compute the partial derivatives of each variable with respect to
/// the coordinates of `DerivativeFrame`.
///
/// \requires `DerivativeTags` to be the head of `VariableTags`
///
/// Returns a `Variables` with a spatial tensor index appended to the front
/// of each tensor within `u` and each `Tag` wrapped with a `Tags::deriv`.
///
/// \tparam DerivativeTags the subset of `VariableTags` for which derivatives
/// are computed.
template <typename DerivativeTags, size_t Dim, typename DerivativeFrame>
void partial_derivatives(
    gsl::not_null<Variables<db::wrap_tags_in<
        Tags::deriv, DerivativeTags, tmpl::size_t<Dim>, DerivativeFrame>>*>
        du,
    const std::array<Variables<DerivativeTags>, Dim>&
        logical_partial_derivatives_of_u,
    const InverseJacobian<DataVector, Dim, Frame::Logical, DerivativeFrame>&
        inverse_jacobian) noexcept;

template <typename DerivativeTags, typename VariableTags, size_t Dim,
          typename DerivativeFrame>
void partial_derivatives(
    gsl::not_null<Variables<db::wrap_tags_in<
        Tags::deriv, DerivativeTags, tmpl::size_t<Dim>, DerivativeFrame>>*>
        du,
    const Variables<VariableTags>& u, const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::Logical, DerivativeFrame>&
        inverse_jacobian) noexcept;

template <typename DerivativeTags, typename VariableTags, size_t Dim,
          typename DerivativeFrame>
auto partial_derivatives(
    const Variables<VariableTags>& u, const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::Logical, DerivativeFrame>&
        inverse_jacobian) noexcept
    -> Variables<db::wrap_tags_in<Tags::deriv, DerivativeTags,
                                  tmpl::size_t<Dim>, DerivativeFrame>>;
// @}

namespace Tags {

/*!
 * \ingroup DataBoxTagsGroup
 * \brief Compute the spatial derivatives of tags in a Variables
 *
 * Computes the spatial derivatives of the Tensors in the Variables represented
 * by `VariablesTag` in the frame mapped to by `InverseJacobianTag`. To only
 * take the derivatives of a subset of these Tensors you can set the
 * `DerivTags` template parameter. It takes a `tmpl::list` of the desired
 * tags and defaults to the full `tags_list` of the Variables.
 *
 * This tag may be retrieved via `db::variables_tag_with_tags_list<VariablesTag,
 * DerivTags>` prefixed with `Tags::deriv`.
 */
template <typename VariablesTag, typename InverseJacobianTag,
          typename DerivTags = typename db::item_type<VariablesTag>::tags_list>
struct DerivCompute
    : db::add_tag_prefix<
          deriv, db::variables_tag_with_tags_list<VariablesTag, DerivTags>,
          tmpl::size_t<tmpl::back<
              typename db::item_type<InverseJacobianTag>::index_list>::dim>,
          typename tmpl::back<
              typename db::item_type<InverseJacobianTag>::index_list>::Frame>,
      db::ComputeTag {
 private:
  using inv_jac_indices =
      typename db::item_type<InverseJacobianTag>::index_list;
  static constexpr auto Dim = tmpl::back<inv_jac_indices>::dim;
  using deriv_frame = typename tmpl::back<inv_jac_indices>::Frame;

 public:
  static constexpr ::Variables<db::wrap_tags_in<
      Tags::deriv, DerivTags, tmpl::size_t<Dim>, deriv_frame>> (*function)(
      const ::Variables<typename db::item_type<VariablesTag>::tags_list>&,
      const ::Mesh<Dim>&,
      const ::InverseJacobian<DataVector, Dim, Frame::Logical, deriv_frame>&) =
      partial_derivatives<DerivTags,
                          typename db::item_type<VariablesTag>::tags_list, Dim,
                          deriv_frame>;
  using argument_tags =
      tmpl::list<VariablesTag, Tags::Mesh<Dim>, InverseJacobianTag>;
};

}  // namespace Tags
