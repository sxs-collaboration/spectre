// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines functions and tags for taking a divergence.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"  // IWYU pragma: keep

/// \cond
class DataVector;
template <size_t Dim>
class Mesh;
template <typename TagsList>
class Variables;

namespace Tags {
template <size_t Dim>
struct Mesh;
/// \endcond

/// \ingroup DataBoxTagsGroup
/// \brief Prefix indicating the divergence
///
/// Prefix indicating the divergence of a Tensor or that a Variables
/// contains divergences of Tensors.
///
/// \snippet Test_Divergence.cpp divergence_name
///
/// \see Tags::DivCompute
template <typename Tag, typename = std::nullptr_t>
struct div;

/// \cond
template <typename Tag>
struct div<Tag, Requires<tt::is_a_v<Tensor, db::item_type<Tag>>>>
    : db::PrefixTag, db::SimpleTag {
  static std::string name() noexcept { return "div(" + Tag::name() + ")"; }
  using tag = Tag;
  using type = TensorMetafunctions::remove_first_index<db::item_type<Tag>>;
};

template <typename Tag>
struct div<Tag, Requires<tt::is_a_v<::Variables, db::item_type<Tag>>>>
    : db::PrefixTag, db::SimpleTag {
  static std::string name() noexcept { return "div(" + Tag::name() + ")"; }
  using tag = Tag;
  using type = db::item_type<Tag>;
};
/// \endcond
}  // namespace Tags

/// \ingroup NumericalAlgorithmsGroup
/// \brief Compute the (Euclidean) divergence of fluxes
template <typename FluxTags, size_t Dim, typename DerivativeFrame>
auto divergence(
    const Variables<FluxTags>& F, const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::Logical, DerivativeFrame>&
        inverse_jacobian) noexcept
    -> Variables<db::wrap_tags_in<Tags::div, FluxTags>>;

namespace Tags {
/// \ingroup DataBoxTagsGroup
/// \brief Compute the divergence of a Variables
///
/// Computes the divergence of the Tensors in the Variables
/// represented by `Tag` in the frame mapped to by
/// `InverseJacobianTag`.  The map must map from the logical frame.
///
/// This tag inherits from `db::add_prefix_tag<Tags::div, Tag>`.
template <typename Tag, typename InverseJacobianTag>
struct DivCompute : db::add_tag_prefix<div, Tag>, db::ComputeTag {
 private:
  using inv_jac_indices =
      typename db::item_type<InverseJacobianTag>::index_list;
  static constexpr auto dim = tmpl::back<inv_jac_indices>::dim;
  static_assert(cpp17::is_same_v<typename tmpl::front<inv_jac_indices>::Frame,
                                 Frame::Logical>,
                "Must map from the logical frame.");

 public:
  static constexpr auto function =
      divergence<typename db::item_type<Tag>::tags_list, dim,
                 typename tmpl::back<inv_jac_indices>::Frame>;
  using argument_tags = tmpl::list<Tag, Tags::Mesh<dim>, InverseJacobianTag>;
};
}  // namespace Tags
