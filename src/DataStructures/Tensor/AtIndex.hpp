// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <concepts>
#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/Metafunctions.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

/*!
 * \brief Get the `Tensor` at a specific grid point index.
 *
 * Works with a `Tensor` of `Kokkos::View`s. Can also be made to work with
 * `DataVector` if needed (use `operator[]` instead of `operator()`).
 */
template <
    typename TensorType, std::integral... Is,
    typename ValueType = typename TensorType::type::value_type,
    typename ResultType = TensorMetafunctions::swap_type<ValueType, TensorType>>
KOKKOS_FUNCTION ResultType make_at_index(const TensorType& tensor,
                                         const Is&... i) {
  ResultType result{};
  for (size_t component = 0; component < TensorType::size(); ++component) {
    result[component] = tensor[component](i...);
  }
  return result;
}

/*!
 * \brief Set the `Tensor` at a specific grid point index.
 *
 * Works with a `Tensor` of `Kokkos::View`s. Can also be made to work with
 * `DataVector` if needed (use `operator[]` instead of `operator()`).
 */
template <typename TensorType, std::integral... Is,
          typename ValueType = typename TensorType::type::value_type>
KOKKOS_FUNCTION void set_at_index(
    const gsl::not_null<TensorType*> tensor,
    const TensorMetafunctions::swap_type<ValueType, TensorType>& value,
    const Is&... i) {
  for (size_t component = 0; component < value.size(); ++component) {
    (*tensor)[component](i...) = value[component];
  }
}

namespace Tags {

/*!
 * \brief Tag representing a specific grid point index of a tensor.
 *
 * This tag replaces the `Tensor` data type with `double` so that it can be
 * used with pointwise operations, e.g. in a `Kokkos::parallel_for` kernel.
 */
template <typename Tag>
struct AtIndex : db::PrefixTag {
 private:
  using TensorType = typename Tag::type;
  using ValueType = typename TensorType::type::value_type;

 public:
  using tag = Tag;
  using type = TensorMetafunctions::swap_type<ValueType, TensorType>;
};

}  // namespace Tags
