// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines functions needed to get Tensor to work with the Intel compiler

#pragma once

#ifdef __INTEL_COMPILER

#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Requires.hpp"

// This is needed because the Intel compiler is not good enough at templates...
namespace Tensor_detail {
template <typename T, typename S, size_t Size>
void increment_tensor_index(std::array<T, Size>& tensor_index,
                            const std::array<S, Size>& dims) {
  for (size_t i = 0; i < Size; ++i) {
    if (++tensor_index[i] < static_cast<T>(dims[i])) {
      break;
    }
    tensor_index[i] = 0;
  }
}

template <size_t I, typename T, typename S, size_t Size,
          Requires<(I == Size - 1)> = nullptr>
constexpr std::array<T, Size> increment_tensor_index_impl(
    const std::array<T, Size>& tensor_index, const std::array<S, Size>& dims) {
  return tensor_index[I] + 1 < static_cast<T>(dims[I])
             ? replace_at<I>(tensor_index, tensor_index[I] + 1)
             : replace_at<I>(tensor_index, static_cast<T>(dims[I]));
}

template <size_t I, typename T, typename S, size_t Size,
          Requires<(I < Size - 1)> = nullptr>
constexpr std::array<T, Size> increment_tensor_index_impl(
    const std::array<T, Size>& tensor_index, const std::array<S, Size>& dims) {
  return tensor_index[I] + 1 < static_cast<T>(dims[I])
             ? replace_at<I>(tensor_index, tensor_index[I] + 1)
             : increment_tensor_index_impl<I + 1>(
                   replace_at<I>(tensor_index, static_cast<T>(dims[I])), dims);
}

template <std::size_t Rank>
constexpr int index_to_swap(const std::array<int, Rank>& arr,
                            const std::array<int, Rank>& sym, int offset,
                            int cur) {
  return Rank == offset
             ? cur
             : ((arr[cur] < arr[offset]) and (sym[cur] == sym[offset]))
                   ? offset
                   : index_to_swap<Rank>(arr, sym, offset + 1, cur);
}

template <typename Symm, typename IndexList, typename T>
T canonicalize_tensor_index(T arr) {
  static constexpr auto symm = ::make_array_from_list<Symm>();
  static constexpr size_t rank = arr.size();
  for (size_t i = 0; i < rank; ++i) {
    std::swap(arr[i], arr[index_to_swap(arr, symm, i, i)]);
  }
  return arr;
}

template <typename Symm, typename IndexList, typename NumComps,
          Requires<(tmpl::size<IndexList>::value > 0)> = nullptr>
std::array<int, NumComps::value> compute_collapsed_to_storage() {
  static constexpr auto dims = ::make_array_from_list<IndexList>();
  static constexpr auto rank = tmpl::size<IndexList>::value;
  std::array<int, NumComps::value> collapsed_to_storage;
  auto tensor_index = make_array<rank>(0);
  int count{0};
  for (auto& current_storage_index : collapsed_to_storage) {
    // Compute canonical tensor_index, which, for symmetric get_tensor_index is
    // in decreasing numerical order, e.g. (3,2) rather than (2,3).
    auto canonical_tensor_index =
        canonicalize_tensor_index<Symm, IndexList>(tensor_index);
    // If the tensor_index was already in the canonical form, then it must be a
    // new unique entry  and we add it to collapsed_to_storage_ as a new
    // integer, thus increasing the size_. Else, the StorageIndex has already
    // been determined so we look it up in the existing collapsed_to_storage
    // table.
    if (tensor_index == canonical_tensor_index) {
      current_storage_index = count;
      ++count;
    } else {
      current_storage_index = collapsed_to_storage[::compute_collapsed_index(
          canonical_tensor_index, dims)];
    }
    // Move to the next tensor_index.
    increment_tensor_index(tensor_index, dims);
  }
  return collapsed_to_storage;
}

template <typename Symm, typename IndexList, typename NumComps,
          Requires<(tmpl::size<IndexList>::value == 0)> = nullptr>
std::array<int, 1> compute_collapsed_to_storage() {
  return std::array<int, 1>{{0}};
}

template <typename Symm, typename IndexList, typename NumIndComps, typename T,
          size_t NumComps,
          Requires<(tmpl::size<IndexList>::value > 0)> = nullptr>
std::array<std::array<int, tmpl::size<IndexList>::value>, NumIndComps::value>
compute_storage_to_tensor(const std::array<T, NumComps>& collapsed_to_storage) {
  static constexpr auto dims = ::make_array_from_list<IndexList>();
  static constexpr auto rank = tmpl::size<IndexList>::value;
  std::array<std::array<int, rank>, NumIndComps::value> storage_to_tensor;
  std::array<int, rank> tensor_index = make_array<rank>(0);
  for (const auto& current_storage_index : collapsed_to_storage) {
    storage_to_tensor[current_storage_index] =
        canonicalize_tensor_index<Symm, IndexList>(tensor_index);
    // Move to the next tensor_index.
    increment_tensor_index(tensor_index, dims);
  }
  return storage_to_tensor;
}

template <typename Symm, typename IndexList, typename NumIndComps, typename T,
          size_t NumComps,
          Requires<(tmpl::size<IndexList>::value == 0)> = nullptr>
std::array<std::array<int, 1>, NumIndComps::value> compute_storage_to_tensor(
    const std::array<T, NumComps>& /*collapsed_to_storage*/) {
  return std::array<std::array<int, 1>, 1>{{std::array<int, 1>{{0}}}};
}

template <typename NumIndComps, typename T, size_t NumComps>
std::array<int, NumIndComps::value> compute_multiplicity(
    const std::array<T, NumComps>& collapsed_to_storage) {
  std::array<int, NumIndComps::value> multiplicity =
      make_array<NumIndComps::value>(0);
  for (const auto& current_storage_index : collapsed_to_storage) {
    ++multiplicity[current_storage_index];
  }
  return multiplicity;
}
}  // namespace Tensor_detail
#endif
