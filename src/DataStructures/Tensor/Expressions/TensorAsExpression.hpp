// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines expressions that represent tensors

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/Tensor/Expressions/LhsTensorSymmAndIndices.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Structure.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

namespace TensorExpressions {
/// \ingroup TensorExpressionsGroup
/// \brief Defines an expression representing a Tensor
///
/// \details
/// In order to represent a tensor as an expression, instead of having Tensor
/// derive off of TensorExpression, a TensorAsExpression derives off of
/// TensorExpression and contains a pointer to a Tensor. The reason having
/// Tensor derive off of TensorExpression is problematic is that the index
/// structure is part of the type of the TensorExpression, so every possible
/// permutation and combination of indices must be derived from. For a rank 3
/// tensor, this is already over 500 base classes, which the Intel compiler
/// takes too long to compile.
///
/// \tparam T the type of Tensor being represented as an expression
/// \tparam ArgsList the tensor indices, e.g. `_a` and `_b` in `F(_a, _b)`
template <typename T, typename ArgsList>
struct TensorAsExpression;

template <typename X, typename Symm, template <typename...> class IndexList,
          typename... Indices, template <typename...> class ArgsList,
          typename... Args>
struct TensorAsExpression<Tensor<X, Symm, IndexList<Indices...>>,
                          ArgsList<Args...>>
    : public TensorExpression<
          TensorAsExpression<Tensor<X, Symm, IndexList<Indices...>>,
                             ArgsList<Args...>>,
          X, Symm, IndexList<Indices...>, ArgsList<Args...>> {
  using type = X;
  using symmetry = Symm;
  using index_list = IndexList<Indices...>;
  static constexpr auto num_tensor_indices = tmpl::size<index_list>::value;
  using args_list = ArgsList<Args...>;
  using structure = Tensor_detail::Structure<symmetry, Indices...>;

  /// Construct an expression from a Tensor
  explicit TensorAsExpression(const Tensor<X, Symm, IndexList<Indices...>>& t)
      : t_(&t) {}
  ~TensorAsExpression() override = default;

  /// \brief Computes the right hand side tensor multi-index that corresponds to
  /// the left hand side tensor multi-index, according to their generic indices
  ///
  /// \details
  /// Given the order of the generic indices for the left hand side (LHS) and
  /// right hand side (RHS) and a specific LHS tensor multi-index, the
  /// computation of the equivalent multi-index for the RHS tensor accounts for
  /// differences in the ordering of the generic indices on the LHS and RHS.
  ///
  /// Here, the elements of `lhs_index_order` and `rhs_index_order` refer to
  /// TensorIndex::values that correspond to generic tensor indices,
  /// `lhs_tensor_multi_index` is a multi-index for the LHS tensor, and the
  /// equivalent RHS tensor multi-index is returned. If we have LHS tensor
  /// \f$L_{ab}\f$, RHS tensor \f$R_{ba}\f$, and the LHS component \f$L_{31}\f$,
  /// the corresponding RHS component is \f$R_{13}\f$.
  ///
  /// Here is an example of what the algorithm does:
  ///
  /// `lhs_index_order`:
  /// \code
  /// [0, 1, 2] // i.e. abc
  /// \endcode
  /// `rhs_index_order`:
  /// \code
  /// [1, 2, 0] // i.e. bca
  /// \endcode
  /// `lhs_tensor_multi_index`:
  /// \code
  /// [4, 0, 3] // i.e. a = 4, b = 0, c = 3
  /// \endcode
  /// returned RHS tensor multi-index:
  /// \code
  /// [0, 3, 4] // i.e. b = 0, c = 3, a = 4
  /// \endcode
  ///
  /// \param lhs_index_order the generic index order of the LHS tensor
  /// \param rhs_index_order the generic index order of the RHS tensor
  /// \param lhs_tensor_multi_index the specific LHS tensor multi-index
  /// \return the RHS tensor multi-index that corresponds to
  /// `lhs_tensor_multi_index`, according to the index orders in
  /// `lhs_index_order` and `rhs_index_order`
  SPECTRE_ALWAYS_INLINE static constexpr std::array<size_t, sizeof...(Indices)>
  compute_rhs_tensor_index(
      const std::array<size_t, sizeof...(Indices)>& lhs_index_order,
      const std::array<size_t, sizeof...(Indices)>& rhs_index_order,
      const std::array<size_t, sizeof...(Indices)>&
          lhs_tensor_multi_index) noexcept {
    std::array<size_t, sizeof...(Indices)> rhs_tensor_multi_index{};
    for (size_t i = 0; i < sizeof...(Indices); ++i) {
      gsl::at(rhs_tensor_multi_index,
              static_cast<unsigned long>(std::distance(
                  rhs_index_order.begin(),
                  alg::find(rhs_index_order, gsl::at(lhs_index_order, i))))) =
          gsl::at(lhs_tensor_multi_index, i);
    }
    return rhs_tensor_multi_index;
  }

  /// \brief Computes a mapping from the storage indices of the left hand side
  /// tensor to the right hand side tensor
  ///
  /// \tparam LhsStructure the Structure of the Tensor on the left hand side of
  /// the TensorExpression
  /// \tparam LhsIndices the TensorIndexs of the Tensor on the left hand side
  /// \return the mapping from the left hand side to the right hand side storage
  /// indices
  template <typename LhsStructure, typename... LhsIndices>
  SPECTRE_ALWAYS_INLINE static constexpr std::array<size_t,
                                                    LhsStructure::size()>
  compute_lhs_to_rhs_map() noexcept {
    constexpr size_t num_components = LhsStructure::size();
    std::array<size_t, num_components> lhs_to_rhs_map{};
    const auto lhs_storage_to_tensor_indices =
        LhsStructure::storage_to_tensor_index();
    for (size_t lhs_storage_index = 0; lhs_storage_index < num_components;
         ++lhs_storage_index) {
      // `compute_rhs_tensor_index` will return the RHS tensor multi-index that
      // corresponds to the LHS tensor multi-index, according to the order of
      // the generic indices for the LHS and RHS. structure::get_storage_index
      // will then get the RHS storage index that corresponds to this RHS
      // tensor multi-index.
      gsl::at(lhs_to_rhs_map, lhs_storage_index) =
          structure::get_storage_index(compute_rhs_tensor_index(
              {{LhsIndices::value...}}, {{Args::value...}},
              lhs_storage_to_tensor_indices[lhs_storage_index]));
    }
    return lhs_to_rhs_map;
  }

  /// \brief Returns the value at a left hand side tensor's storage index
  ///
  /// \details
  /// One big challenge with TensorExpression implementation is the reordering
  /// of the indices on the left hand side (LHS) and right hand side (RHS) of
  /// the expression. The algorithms implemented in `compute_lhs_to_rhs_map` and
  /// `compute_rhs_tensor_index` handle the index sorting by mapping between the
  /// generic index orders of the LHS and RHS.
  ///
  /// \tparam LhsStructure the Structure of the Tensor on the LHS of the
  /// TensorExpression
  /// \tparam LhsIndices the TensorIndexs of the Tensor on the LHS of the tensor
  /// expression
  /// \param lhs_storage_index the storage index of the LHS tensor component to
  /// retrieve
  /// \return the value of the DataType of the component at `lhs_storage_index`
  /// in the LHS tensor
  template <typename LhsStructure, typename... LhsIndices>
  SPECTRE_ALWAYS_INLINE decltype(auto) get(
      const size_t lhs_storage_index) const noexcept {
    if constexpr (std::is_same_v<LhsStructure, structure> and
                  std::is_same_v<tmpl::list<LhsIndices...>,
                                 tmpl::list<Args...>>) {
      return (*t_)[lhs_storage_index];
    } else {
      constexpr std::array<size_t, LhsStructure::size()> lhs_to_rhs_map =
          compute_lhs_to_rhs_map<LhsStructure, LhsIndices...>();
      return (*t_)[gsl::at(lhs_to_rhs_map, lhs_storage_index)];
    }
  }

  /// Retrieve the i'th entry of the Tensor being held
  SPECTRE_ALWAYS_INLINE type operator[](const size_t i) const {
    return t_->operator[](i);
  }

 private:
  const Tensor<X, Symm, IndexList<Indices...>>* t_ = nullptr;
};
}  // namespace TensorExpressions
