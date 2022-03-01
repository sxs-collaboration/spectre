// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/TMPL.hpp"

namespace tenex {
/// \ingroup TensorExpressionsGroup
/// \brief Defines an expression representing a `double`
struct NumberAsExpression
    : public TensorExpression<NumberAsExpression, double, tmpl::list<>,
                              tmpl::list<>, tmpl::list<>> {
  // === Index properties ===
  /// The type of the data being stored in the result of the expression
  using type = double;
  /// The list of \ref SpacetimeIndex "TensorIndexType"s of the result of the
  /// expression
  using symmetry = tmpl::list<>;
  /// The list of \ref SpacetimeIndex "TensorIndexType"s of the result of the
  /// expression
  using index_list = tmpl::list<>;
  /// The list of generic `TensorIndex`s of the result of the expression
  using args_list = tmpl::list<>;
  /// The number of tensor indices in the result of the expression
  static constexpr auto num_tensor_indices = 0;

  // === Arithmetic tensor operations properties ===
  /// The number of arithmetic tensor operations done in the subtree for the
  /// left operand, which is 0 because this is a leaf expression
  static constexpr size_t num_ops_left_child = 0;
  /// The number of arithmetic tensor operations done in the subtree for the
  /// right operand, which is 0 because this is a leaf expression
  static constexpr size_t num_ops_right_child = 0;
  /// The total number of arithmetic tensor operations done in this expression's
  /// whole subtree, which is 0 because this is a leaf expression
  static constexpr size_t num_ops_subtree = 0;

  NumberAsExpression(const double number) : number_(number) {}
  ~NumberAsExpression() override = default;

  /// \brief Returns the number represented by the expression
  ///
  /// \return the number represented by this expression
  SPECTRE_ALWAYS_INLINE double get(
      const std::array<size_t, num_tensor_indices>& /*multi_index*/) const {
    return number_;
  }

 private:
  /// Number represented by this expression
  double number_;
};
}  // namespace tenex
