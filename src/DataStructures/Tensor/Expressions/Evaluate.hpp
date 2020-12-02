// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines function TensorExpressions::evaluate(TensorExpression)

#pragma once

#include "DataStructures/Tensor/Expressions/LhsTensorSymmAndIndices.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Requires.hpp"

namespace TensorExpressions {
/*!
 * \ingroup TensorExpressionsGroup
 * \brief Evaluate a RHS tensor expression to a tensor with the LHS index order
 * set in the template parameters
 *
 * \details Uses the right hand side (RHS) TensorExpression's index ordering
 * (`T::args_list`) and the desired left hand side (LHS) tensor's index ordering
 * (`LhsTensorIndices`) to construct a LHS Tensor with that LHS index ordering.
 * This can carry out the evaluation of a RHS tensor expression to a LHS tensor
 * with the same index ordering, such as \f$L_{ab} = R_{ab}\f$, or different
 * ordering, such as \f$L_{ba} = R_{ab}\f$.
 *
 * ### Example usage
 * Given two rank 2 Tensors `R` and `S` with index order (a, b), add them
 * together and generate the resultant LHS Tensor `L` with index order (b, a):
 * \code{.cpp}
 * auto L = TensorExpressions::evaluate<ti_b_t, ti_a_t>(
 *     R(ti_a, ti_b) + S(ti_a, ti_b));
 * \endcode
 * \metareturns Tensor
 *
 * This represents evaluating: \f$L_{ba} = \R_{ab} + S_{ab}\f$
 *
 * @tparam LhsTensorIndices the TensorIndex of the Tensor on the LHS of the
 * tensor expression, e.g. `ti_a_t`, `ti_b_t`, `ti_c_t`
 * @tparam T the type of the RHS TensorExpression
 * @param rhs_te the RHS TensorExpression to be evaluated
 * @return the LHS Tensor with index order specified by LhsTensorIndices
 */
template <typename... LhsTensorIndices, typename T,
          Requires<std::is_base_of<Expression, T>::value> = nullptr>
auto evaluate(const T& rhs_te) {
  static_assert(
      sizeof...(LhsTensorIndices) == tmpl::size<typename T::args_list>::value,
      "Must have the same number of indices on the LHS and RHS of a tensor "
      "equation.");
  using rhs = tmpl::transform<tmpl::remove_duplicates<typename T::args_list>,
                              std::decay<tmpl::_1>>;
  static_assert(
      tmpl::equal_members<tmpl::list<std::decay_t<LhsTensorIndices>...>,
                          rhs>::value,
      "All indices on the LHS of a Tensor Expression (that is, those specified "
      "in evaluate<Indices::...>) must be present on the RHS of the expression "
      "as well.");

  using rhs_tensorindex_list = typename T::args_list;
  using lhs_tensorindex_list = tmpl::list<LhsTensorIndices...>;
  using rhs_symmetry = typename T::symmetry;
  using rhs_tensorindextype_list = typename T::index_list;

  // Stores (potentially reordered) symmetry and indices needed for constructing
  // the LHS tensor with index order specified by LhsTensorIndices
  using lhs_tensor =
      LhsTensorSymmAndIndices<rhs_tensorindex_list, lhs_tensorindex_list,
                              rhs_symmetry, rhs_tensorindextype_list>;

  // Construct and return LHS tensor
  return Tensor<typename T::type, typename lhs_tensor::symmetry,
                typename lhs_tensor::tensorindextype_list>(
      rhs_te, tmpl::list<LhsTensorIndices...>{});
}
}  // namespace TensorExpressions
