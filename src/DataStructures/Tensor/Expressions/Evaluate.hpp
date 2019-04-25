// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines function TensorExpressions::evaluate(TensorExpression)

#pragma once

#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Requires.hpp"

namespace TensorExpressions {

/*!
 * \ingroup TensorExpressionsGroup
 * \brief Evaluate a Tensor Expression with LHS indices set in the template
 * parameters
 *
 * @tparam LhsIndices the indices on the left hand side of the tensor expression
 * @return Tensor<typename T::type, typename T::symmetry, typename
 * T::index_list>
 */
template <typename... LhsIndices, typename T,
          Requires<std::is_base_of<Expression, T>::value> = nullptr>
auto evaluate(const T& te) {
  static_assert(
      sizeof...(LhsIndices) == tmpl::size<typename T::args_list>::value,
      "Must have the same number of indices on the LHS and RHS of a tensor "
      "equation.");
  using rhs = tmpl::transform<tmpl::remove_duplicates<typename T::args_list>,
                              std::decay<tmpl::_1>>;
  static_assert(
      tmpl::equal_members<tmpl::list<std::decay_t<LhsIndices>...>, rhs>::value,
      "All indices on the LHS of a Tensor Expression (that is, those specified "
      "in evaluate<Indices::...>) must be present on the RHS of the expression "
      "as well.");
  return Tensor<typename T::type, typename T::symmetry, typename T::index_list>(
      te, tmpl::list<LhsIndices...>{});
}

}  // namespace TensorExpressions
