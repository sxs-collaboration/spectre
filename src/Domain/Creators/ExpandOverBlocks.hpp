// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace domain {
/*!
 * \brief Produce a distribution of type `T` over all blocks and dimensions in
 * the domain, based on values `T` of variable isotropy and homogeneity.
 *
 * This class is useful to option-create values for e.g. the initial refinement
 * level or initial number of grid points for domain creators. It can be used
 * with `std::visit` and a `std::variant` with (a subset of) these types:
 *
 * - `T`: Repeat over all blocks and dimensions (isotropic and homogeneous).
 * - `std::array<T, Dim>`: Repeat over all blocks (homogeneous).
 * - `std::vector<std::array<T, Dim>>>`: Only check if the size matches the
 *   number of blocks, throwing a `std::length_error` if it doesn't.
 * - `std::unordered_map<std::string, std::array<T, Dim>>`: Map block names, or
 *   names of block groups, to values. The map must cover all blocks once the
 *   groups are expanded. To use this option you must pass the list of block
 *   names and groups to the constructor.
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
 * \tparam Dim The number of spatial dimensions
 */
template <typename T, size_t Dim>
struct ExpandOverBlocks {
  ExpandOverBlocks(size_t num_blocks) noexcept;
  ExpandOverBlocks(
      std::vector<std::string> block_names,
      std::unordered_map<std::string, std::unordered_set<std::string>>
          block_groups = {}) noexcept;

  /// Repeat over all blocks and dimensions (isotropic and homogeneous)
  std::vector<std::array<T, Dim>> operator()(const T& value) const;

  /// Repeat over all blocks (homogeneous)
  std::vector<std::array<T, Dim>> operator()(
      const std::array<T, Dim>& value) const;

  /// Only check if the size matches the number of blocks, throwing a
  /// `std::length_error` if it doesn't
  std::vector<std::array<T, Dim>> operator()(
      std::vector<std::array<T, Dim>> value) const;

  /// Map block names, or names of block groups, to values. The map must cover
  /// all blocks once the groups are expanded. To use this option you must pass
  /// the list of block names and groups to the constructor. Here's an example:
  ///
  /// \snippet Test_ExpandOverBlocks.cpp expand_over_blocks_named_example
  std::vector<std::array<T, Dim>> operator()(
      const std::unordered_map<std::string, std::array<T, Dim>>& value) const;

 private:
  size_t num_blocks_;
  std::vector<std::string> block_names_;
  std::unordered_map<std::string, std::unordered_set<std::string>>
      block_groups_;
};
}  // namespace domain
