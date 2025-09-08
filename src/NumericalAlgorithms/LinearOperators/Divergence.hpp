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
#include "Utilities/TypeTraits.hpp"
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
auto divergence(const Variables<FluxTags>& F, const Mesh<Dim>& mesh,
                const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                                      DerivativeFrame>& inverse_jacobian)
    -> Variables<db::wrap_tags_in<Tags::div, FluxTags>>;

template <typename... DivTags, typename... FluxTags, size_t Dim,
          typename DerivativeFrame>
void divergence(
    gsl::not_null<Variables<tmpl::list<DivTags...>>*> divergence_of_F,
    const Variables<tmpl::list<FluxTags...>>& F, const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          DerivativeFrame>& inverse_jacobian);
/// @}

/// @{
/// \ingroup NumericalAlgorithmsGroup
/// \brief Compute the divergence of the vector `input`
template <typename DataType, size_t Dim, typename DerivativeFrame>
Scalar<DataType> divergence(
    const tnsr::I<DataType, Dim, DerivativeFrame>& input, const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          DerivativeFrame>& inverse_jacobian);

template <typename DataType, size_t Dim, typename DerivativeFrame>
void divergence(gsl::not_null<Scalar<DataType>*> div_input,
                const tnsr::I<DataType, Dim, DerivativeFrame>& input,
                const Mesh<Dim>& mesh,
                const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                                      DerivativeFrame>& inverse_jacobian);
/// @}

/// @{
/*!
 * \ingroup NumericalAlgorithmsGroup
 * \brief Compute the divergence of fluxes where a Cartoon basis is being
 * utilized.
 *
 * The additional parameter `inertial_coords` is used for division by the
 * \f$x\f$ coordinates. If \f$x=0\f$ is included in the domain, it is assumed to
 * be present only at the first index and is handled by L'H&ocirc;pital's rule.
 *
 * The mesh is required to have the Cartoon basis in the last and potentially
 * second-to-last coordinates and the inverse jacobian is accordingly used only
 * in the first and potentially second dimensions.
 *
 * \see cartoon_partial_derivatives for details on Cartoon derivatives.
 */
template <typename... DivTags, typename... FluxTags, size_t Dim,
          typename DerivativeFrame, Requires<Dim == 3> = nullptr>
void cartoon_divergence(
    gsl::not_null<Variables<tmpl::list<DivTags...>>*> divergence_of_F,
    const Variables<tmpl::list<FluxTags...>>& F, const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          DerivativeFrame>& inverse_jacobian_3d,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& inertial_coords);
/// @}

/// @{
/*!
 * \ingroup NumericalAlgorithmsGroup
 * \brief Calls the correct divergence function, either normal divergence or
 * cartoon divergence, as determined by mesh basis.
 *
 * If you have a `Variables` with several tensors with Cartoon bases you need
 * to find the divergence of, you should use the `divergence` function
 * that operates on `Variables` since that'll be more efficient.
 */
template <typename... DivTags, typename... FluxTags, size_t Dim,
          typename DerivativeFrame>
void divergence(
    gsl::not_null<Variables<tmpl::list<DivTags...>>*> div_fluxes,
    const Variables<tmpl::list<FluxTags...>>& fluxes, const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          DerivativeFrame>& inverse_jacobian_3d,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& inertial_coords);
template <typename DataType, size_t Dim, typename DerivativeFrame>
void divergence(
    gsl::not_null<Scalar<DataType>*> div_input,
    const tnsr::I<DataType, Dim, DerivativeFrame>& input, const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          DerivativeFrame>& inverse_jacobian,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& inertial_coords);
template <typename DataType, size_t Dim, typename DerivativeFrame>
Scalar<DataType> divergence(
    const tnsr::I<DataType, Dim, DerivativeFrame>& input, const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          DerivativeFrame>& inverse_jacobian,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& inertial_coords);
/// @}

/// @{
/*!
 * \brief Compute the divergence of fluxes in logical coordinates
 *
 * Applies the logical differentiation matrix to the fluxes in each dimension
 * and sums over dimensions.
 *
 * \see divergence
 */
template <typename ResultTensor, typename FluxTensor, size_t Dim>
void logical_divergence(gsl::not_null<ResultTensor*> div_flux,
                        const FluxTensor& flux, const Mesh<Dim>& mesh);

template <typename FluxTags, size_t Dim>
auto logical_divergence(const Variables<FluxTags>& flux, const Mesh<Dim>& mesh)
    -> Variables<db::wrap_tags_in<Tags::div, FluxTags>>;

template <typename... DivTags, typename... FluxTags, size_t Dim>
void logical_divergence(
    gsl::not_null<Variables<tmpl::list<DivTags...>>*> div_flux,
    const Variables<tmpl::list<FluxTags...>>& flux, const Mesh<Dim>& mesh);
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
template <typename Tag, typename MeshTag, typename InverseJacobianTag>
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
      const Mesh<dim>&, const typename InverseJacobianTag::type&) = &divergence;
  using argument_tags =
      tmpl::list<Tag, domain::Tags::Mesh<dim>, InverseJacobianTag>;
};

/// \ingroup DataBoxTagsGroup
/// \brief Compute the divergence of a `tnsr::I` (vector)
///
/// This tag inherits from `db::add_tag_prefix<Tags::div, Tag>`.
///
/// For an executable that does not allow a Cartoon basis, the last parameter,
/// `InertialCoordsTag`, should not be passed.
template <typename Tag, typename MeshTag, typename InverseJacobianTag,
          typename InertialCoordsTag = void>
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
  static constexpr void function(
      const gsl::not_null<return_type*> div_input,
      const typename Tag::type& input, const Mesh<dim>& mesh,
      const typename InverseJacobianTag::type& inverse_jacobian) {
    divergence(div_input, input, mesh, inverse_jacobian);
  }
  static constexpr void function(
      const gsl::not_null<return_type*> div_input,
      const typename Tag::type& input, const Mesh<dim>& mesh,
      const typename InverseJacobianTag::type& inverse_jacobian,
      const tnsr::I<DataVector, dim, Frame::Inertial>& inertial_coords) {
    divergence(div_input, input, mesh, inverse_jacobian, inertial_coords);
  }
  using argument_tags = tmpl::conditional_t<
      std::is_same_v<void, InertialCoordsTag>,
      tmpl::list<Tag, MeshTag, InverseJacobianTag>,
      tmpl::list<Tag, MeshTag, InverseJacobianTag, InertialCoordsTag>>;
};
}  // namespace Tags
