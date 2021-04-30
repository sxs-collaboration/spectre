// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Utilities/MakeArray.hpp"

namespace domain {

namespace ExpandOverBlocks_detail {
template <typename T, typename U, typename = std::void_t<>>
struct is_value_type {
  static constexpr bool value = false;
};

template <typename T, typename U>
struct is_value_type<T, U, std::void_t<typename T::value_type>> {
  static constexpr bool value = std::is_same_v<U, typename T::value_type>;
};
}  // namespace ExpandOverBlocks_detail

/*!
 * \brief Produce a std::vector<T> over all blocks of the domain
 *
 * This class is useful to option-create values for e.g. the initial refinement
 * level or initial number of grid points for domain creators. It can be used
 * with `std::visit` and a `std::variant` with (a subset of) these types:
 *
 * - `T`: Repeat the given value over all blocks (homogeneous).
 * - `std::vector<T>`: Only check if the size matches the
 *   number of blocks, throwing a `std::length_error` if it doesn't.
 * - `std::unordered_map<std::string, T>`: Map block names, or
 *   names of block groups, to values. The map must cover all blocks once the
 *   groups are expanded. To use this option you must pass the list of block
 *   names and groups to the constructor.
 * - `T::value_type`: Repeat the given value over all blocks and dimensions
 *   (isotropic and homogeneous). Only works if `T` is a `std::array`. For
 *   example, if `T` is `std::array<size_t, 3>`, this will produce a
 *   `std::vector<std::array<size_t, 3>>` with the same value repeated
 *   `num_blocks x 3` times.
 *
 * If `T` is a `std::unique_ptr`, the class will clone the value for each block
 * using `T::get_clone()`.
 *
 * Note that the call-operators `throw` when they encounter errors, such as
 * mismatches in the number of blocks. The exceptions can be used to output
 * user-facing error messages in an option-parsing context.
 *
 * Here's an example for using this class:
 *
 * \snippet Test_ExpandOverBlocks.cpp expand_over_blocks_example
 *
 * Here's an example using block names and groups:
 *
 * \snippet Test_ExpandOverBlocks.cpp expand_over_blocks_named_example
 *
 * \tparam T The type distributed over the domain
 */
template <typename T>
struct ExpandOverBlocks {
  ExpandOverBlocks(size_t num_blocks);
  ExpandOverBlocks(
      std::vector<std::string> block_names,
      std::unordered_map<std::string, std::unordered_set<std::string>>
          block_groups = {});

  /// Repeat over all blocks (homogeneous)
  std::vector<T> operator()(const T& value) const;

  /// Only check if the size matches the number of blocks, throwing a
  /// `std::length_error` if it doesn't
  std::vector<T> operator()(const std::vector<T>& value) const;

  /// Map block names, or names of block groups, to values. The map must cover
  /// all blocks once the groups are expanded. To use this option you must pass
  /// the list of block names and groups to the constructor. Here's an example:
  ///
  /// \snippet Test_ExpandOverBlocks.cpp expand_over_blocks_named_example
  std::vector<T> operator()(
      const std::unordered_map<std::string, T>& value) const;

  /// Repeat over all blocks and dimensions (isotropic and homogeneous)
  template <
      typename U,
      Requires<ExpandOverBlocks_detail::is_value_type<T, U>::value> = nullptr>
  std::vector<T> operator()(const U& value) const {
    return {num_blocks_, make_array<std::tuple_size_v<T>>(value)};
  }

 private:
  size_t num_blocks_;
  std::vector<std::string> block_names_;
  std::unordered_map<std::string, std::unordered_set<std::string>>
      block_groups_;
};

}  // namespace domain
