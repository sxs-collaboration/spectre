// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

namespace TensorExpressions {
/// \ingroup TensorExpressionsGroup
/// \brief Defines an expression representing a `double`
struct NumberAsExpression
    : public TensorExpression<NumberAsExpression, double, tmpl::list<>,
                              tmpl::list<>, tmpl::list<>> {
  using type = double;
  using symmetry = tmpl::list<>;
  using index_list = tmpl::list<>;
  using args_list = tmpl::list<>;
  static constexpr auto num_tensor_indices = 0;

  NumberAsExpression(const double number) : number_(number) {}
  ~NumberAsExpression() override = default;

  /// \brief Returns the number represented by the expression
  ///
  /// \details
  /// While a NumberAsExpression does not store a rank 0 Tensor, it does
  /// represent one. This is why this template is only defined for the case
  /// where `TensorIndices` is empty.
  ///
  /// \tparam TensorIndices the TensorIndexs of the LHS tensor and RHS tensor
  /// expression
  /// \return the number represented by this expression
  template <typename... TensorIndices,
            Requires<sizeof...(TensorIndices) == 0> = nullptr>
  SPECTRE_ALWAYS_INLINE double get(
      const std::array<size_t, 0>& /*multi_index*/) const noexcept {
    return number_;
  }

 private:
  double number_;
};
}  // namespace TensorExpressions
