// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines metafunctions used to comute the Symmetry<...> for a Tensor

#pragma once

#include <type_traits>

#include "Utilities/Array.hpp"
#include "Utilities/TMPL.hpp"

namespace detail {
template <size_t Size>
constexpr int find_reduced_index(
    const cpp20::array<std::pair<int, int>, Size>& t,
    const int value) noexcept {
  for (size_t i = 0; i < Size; ++i) {
    if (t[i].first == value) {
      return t[i].second;
    }
  }
  return 0;
}

template <size_t Size>
constexpr cpp20::array<int, Size> symmetry(
    const std::array<int, Size>& input_symm) noexcept {
  cpp20::array<int, Size> output_symm{};
  int next_symm_entry = 1;
  cpp20::array<std::pair<int, int>, Size> input_to_output_map{};
  size_t input_to_output_map_size = 0;
  for (size_t i = Size - 1; i < Size; --i) {
    // clang-tidy: use gsl::at
    int found_symm_entry =
        find_reduced_index(input_to_output_map, input_symm[i]);  // NOLINT
    if (found_symm_entry == 0) {
      output_symm[i] = next_symm_entry;  // NOLINT
      input_to_output_map[input_to_output_map_size].first =
          input_symm[i];  // NOLINT
      input_to_output_map[input_to_output_map_size].second = output_symm[i];
      input_to_output_map_size++;
      next_symm_entry++;
    } else {
      output_symm[i] = found_symm_entry;
    }
  }
  return output_symm;
}

template <typename IndexSequence, typename SymmetrySequence>
struct SymmetryImpl;

template <size_t... Is, std::int32_t... Ss>
struct SymmetryImpl<std::index_sequence<Is...>,
                    tmpl::integral_list<std::int32_t, Ss...>> {
  static_assert((... and (Ss > 0)),
                "Symmetry values must be positive integers.");
  static constexpr cpp20::array<int, sizeof...(Is)> t =
      symmetry(std::array<int, sizeof...(Is)>{{Ss...}});
  using type = tmpl::integral_list<std::int32_t, t[Is]...>;
};
}  // namespace detail

/// \ingroup TensorGroup
/// \brief Computes the canonical symmetry from the integers `T`
///
/// \details
/// Compute the canonical symmetry typelist given a set of integers, T. The
/// resulting typelist is in ascending order of the integers, from right to
/// left. For example, the result of `Symmetry<1, 2, 1, 3>` is
/// `integral_list<int32_t, 2, 3, 2, 1>`. Anti-symmetries are not currently
/// supported.
///
/// \tparam T the integers denoting the symmetry of the Tensor
template <std::int32_t... T>
using Symmetry = typename detail::SymmetryImpl<
    std::make_index_sequence<sizeof...(T)>,
    tmpl::integral_list<std::int32_t, T...>>::type;
