// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Domain/Creators/ExpandOverBlocks.hpp"

#include <boost/algorithm/string/join.hpp>
#include <cstddef>
#include <exception>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Utilities/CloneUniquePtrs.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

namespace domain {

template <typename T>
ExpandOverBlocks<T>::ExpandOverBlocks(size_t num_blocks)
    : num_blocks_(num_blocks) {}

template <typename T>
ExpandOverBlocks<T>::ExpandOverBlocks(
    std::vector<std::string> block_names,
    std::unordered_map<std::string, std::unordered_set<std::string>>
        block_groups)
    : num_blocks_(block_names.size()),
      block_names_(std::move(block_names)),
      block_groups_(std::move(block_groups)) {}

template <typename T>
std::vector<T> ExpandOverBlocks<T>::operator()(const T& value) const {
  if constexpr (tt::is_a_v<std::unique_ptr, T>) {
    std::vector<T> expanded(num_blocks_);
    for (size_t i = 0; i < num_blocks_; ++i) {
      expanded[i] = value->get_clone();
    }
    return expanded;
  } else {
    return {num_blocks_, value};
  }
}

template <typename T>
std::vector<T> ExpandOverBlocks<T>::operator()(
    const std::vector<T>& value) const {
  if (value.size() != num_blocks_) {
    throw std::length_error{"You supplied " + std::to_string(value.size()) +
                            " values, but the domain creator has " +
                            std::to_string(num_blocks_) + " blocks."};
  }
  if constexpr (tt::is_a_v<std::unique_ptr, T>) {
    return clone_unique_ptrs(value);
  } else {
    return value;
  }
}

template <typename T>
std::vector<T> ExpandOverBlocks<T>::operator()(
    const std::unordered_map<std::string, T>& value) const {
  ASSERT(num_blocks_ == block_names_.size(),
         "Construct 'ExpandOverBlocks' with block names to use the "
         "map-over-block-names feature.");
  // Expand group names
  auto value_per_block = [&value]() {
    if constexpr (tt::is_a_v<std::unique_ptr, T>) {
      return clone_unique_ptrs(value);
    } else {
      return value;
    }
  }();
  for (const auto& [name, block_value] : value) {
    const auto found_group = block_groups_.find(name);
    if (found_group != block_groups_.end()) {
      for (const auto& expanded_name : found_group->second) {
        if (value_per_block.count(expanded_name) == 0) {
          value_per_block[expanded_name] = [&local_block_value =
                                                block_value]() {
            if constexpr (tt::is_a_v<std::unique_ptr, T>) {
              return local_block_value->get_clone();
            } else {
              return local_block_value;
            }
          }();
        } else {
          throw std::invalid_argument{
              "Duplicate block name '" + expanded_name +
              // NOLINTNEXTLINE(performance-inefficient-string-concatenation)
              "' (expanded from '" + name + "')."};
        }
      }
      value_per_block.erase(name);
    }
  }
  if (value_per_block.size() != num_blocks_) {
    throw std::length_error{
        "You supplied " + std::to_string(value_per_block.size()) +
        " values, but the domain creator has " + std::to_string(num_blocks_) +
        " blocks: " + boost::algorithm::join(block_names_, ", ")};
  }
  std::vector<T> result{};
  result.reserve(num_blocks_);
  for (const auto& block_name : block_names_) {
    const auto found_value = value_per_block.find(block_name);
    if (found_value != value_per_block.end()) {
      result.emplace_back(std::move(found_value->second));
    } else {
      throw std::out_of_range{"Value for block '" + block_name +
                              "' is missing. Did you misspell its name?"};
    }
  }
  return result;
}

}  // namespace domain
