// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines functions and tags for taking a divergence.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"  // IWYU pragma: keep
#include "Utilities/TypeTraits/IsA.hpp"

/// \cond
class DataVector;
template <size_t Dim>
class Mesh;
template <typename TagsList>
class Variables;

namespace domain {
namespace Tags {
template <size_t Dim>
struct Mesh;
}  // namespace Tags
}  // namespace domain
/// \endcond

namespace Tags {
/// \ingroup DataBoxTagsGroup
/// \brief Prefix indicating the divergence
///
/// Prefix indicating the divergence of a Tensor.
///
/// \see Tags::DivVectorCompute Tags::DivVariablesCompute
template <typename Tag, typename = std::nullptr_t>
struct div;

/// \cond
template <typename Tag>
struct div<Tag, Requires<tt::is_a_v<Tensor, typename Tag::type>>>
    : db::PrefixTag, db::SimpleTag {
  using tag = Tag;
  using type = TensorMetafunctions::remove_first_index<typename Tag::type>;
};
/// \endcond
}  // namespace Tags

/// @{
/// \ingroup NumericalAlgorithmsGroup
/// \brief Compute the (Euclidean) divergence of fluxes
template <typename FluxTags, size_t Dim, typename DerivativeFrame>
auto divergence(
    const Variables<FluxTags>& F, const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          DerivativeFrame>& inverse_jacobian) noexcept
    -> Variables<db::wrap_tags_in<Tags::div, FluxTags>>;

template <typename... DivTags, typename... FluxTags, size_t Dim,
          typename DerivativeFrame>
void divergence(
    gsl::not_null<Variables<tmpl::list<DivTags...>>*> divergence_of_F,
    const Variables<tmpl::list<FluxTags...>>& F, const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          DerivativeFrame>& inverse_jacobian) noexcept;
/// @}

/// @{
/// \ingroup NumericalAlgorithmsGroup
/// \brief Compute the divergence of the vector `input`
template <size_t Dim, typename DerivativeFrame>
Scalar<DataVector> divergence(
    const tnsr::I<DataVector, Dim, DerivativeFrame>& input,
    const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          DerivativeFrame>& inverse_jacobian) noexcept;

template <size_t Dim, typename DerivativeFrame>
void divergence(
    gsl::not_null<Scalar<DataVector>*> div_input,
    const tnsr::I<DataVector, Dim, DerivativeFrame>& input,
    const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          DerivativeFrame>& inverse_jacobian) noexcept;
/// @}

namespace Tags {
/*!
 * \ingroup DataBoxTagsGroup
 * \brief Compute the divergence of a Variables
 *
 * Computes the divergence of the every Tensor in the Variables represented by
 * `Tag`. The first index of each Tensor must be an upper spatial index, i.e.,
 * the first index must have type
 * `TensorIndexType<Dim, UpLo::Up, Frame::TargetFrame, IndexType::Spatial>`.
 * The divergence is computed in the frame `TargetFrame`, and
 * `InverseJacobianTag` must be associated with a map from
 * `Frame::ElementLogical` to `Frame::TargetFrame`.
 *
 * Note that each tensor may have additional tensor indices - in this case the
 * divergence is computed for each additional index. For instance, a tensor
 * \f$F^i_{ab}\f$ has divergence
 * \f$Div_{ab} = \partial_i F^i_{ab}\f$. This is to accommodate evolution
 * equations where the evolved variables \f$u_\alpha\f$ are higher-rank tensors
 * and thus their fluxes can be written as \f$F^i_\alpha\f$. A simple example
 * would be the fluid velocity in hydro systems, where we would write the flux
 * as \f$F^{ij}\f$.
 *
 * This tag inherits from `db::add_tag_prefix<Tags::div, Tag>`.
 */
template <typename Tag, typename InverseJacobianTag>
struct DivVariablesCompute : db::add_tag_prefix<div, Tag>, db::ComputeTag {
 private:
  using inv_jac_indices = typename InverseJacobianTag::type::index_list;
  static constexpr auto dim = tmpl::back<inv_jac_indices>::dim;
  static_assert(std::is_same_v<typename tmpl::front<inv_jac_indices>::Frame,
                               Frame::ElementLogical>,
                "Must map from the logical frame.");

 public:
  using base = db::add_tag_prefix<div, Tag>;
  using return_type = typename base::type;
  static constexpr void (*function)(
      const gsl::not_null<return_type*>, const typename Tag::type&,
      const Mesh<dim>&, const typename InverseJacobianTag::type&) = divergence;
  using argument_tags =
      tmpl::list<Tag, domain::Tags::Mesh<dim>, InverseJacobianTag>;
};

/// \ingroup DataBoxTagsGroup
/// \brief Compute the divergence of a `tnsr::I` (vector)
///
/// This tag inherits from `db::add_tag_prefix<Tags::div, Tag>`.
template <typename Tag, typename InverseJacobianTag>
struct DivVectorCompute : div<Tag>, db::ComputeTag {
 private:
  using inv_jac_indices = typename InverseJacobianTag::type::index_list;
  static constexpr auto dim = tmpl::back<inv_jac_indices>::dim;
  static_assert(std::is_same_v<typename tmpl::front<inv_jac_indices>::Frame,
                               Frame::ElementLogical>,
                "Must map from the logical frame.");

 public:
  using base = div<Tag>;
  using return_type = typename base::type;
  static constexpr void (*function)(const gsl::not_null<return_type*>,
                                    const typename Tag::type&, const Mesh<dim>&,
                                    const typename InverseJacobianTag::type&) =
      divergence<dim, typename tmpl::back<inv_jac_indices>::Frame>;
  using argument_tags =
      tmpl::list<Tag, domain::Tags::Mesh<dim>, InverseJacobianTag>;
};
}  // namespace Tags
