// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines the tensor expression representing negation

#pragma once

#include <array>
#include <cstddef>
#include <utility>

#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace TensorExpressions {
/// \ingroup TensorExpressionsGroup
/// \brief Defines the tensor expression representing the negation of a tensor
/// expression
///
/// \tparam T the type of tensor expression being negated
template <typename T>
struct Negate
    : public TensorExpression<Negate<T>, typename T::type, typename T::symmetry,
                              typename T::index_list, typename T::args_list> {
  using type = typename T::type;
  using symmetry = typename T::symmetry;
  using index_list = typename T::index_list;
  using args_list = typename T::args_list;
  static constexpr auto num_tensor_indices = tmpl::size<index_list>::value;

  Negate(T t) : t_(std::move(t)) {}
  ~Negate() override = default;

  /// \brief Return the value of the component of the negated tensor expression
  /// at a given multi-index
  ///
  /// \param multi_index the multi-index of the component to retrieve from the
  /// negated tensor expression
  /// \return the value of the component at `multi_index` in the negated tensor
  /// expression
  SPECTRE_ALWAYS_INLINE decltype(auto) get(
      const std::array<size_t, num_tensor_indices>& multi_index) const {
    return -t_.get(multi_index);
  }

 private:
  T t_;
};
}  // namespace TensorExpressions

/// \ingroup TensorExpressionsGroup
/// \brief Returns the tensor expression representing the negation of a tensor
/// expression
///
/// \param t the tensor expression
/// \return the tensor expression representing the negation of `t`
template <typename T>
SPECTRE_ALWAYS_INLINE auto operator-(
    const TensorExpression<T, typename T::type, typename T::symmetry,
                           typename T::index_list, typename T::args_list>& t) {
  return TensorExpressions::Negate<T>(~t);
}
