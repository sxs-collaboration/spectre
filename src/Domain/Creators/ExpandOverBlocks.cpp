// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/ExpandOverBlocks.hpp"

#include <array>
#include <boost/algorithm/string/join.hpp>
#include <cstddef>
#include <exception>
#include <string>
#include <vector>

#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeArray.hpp"

namespace domain {

template <typename T, size_t Dim>
ExpandOverBlocks<T, Dim>::ExpandOverBlocks(size_t num_blocks)
    : num_blocks_(num_blocks) {}

template <typename T, size_t Dim>
ExpandOverBlocks<T, Dim>::ExpandOverBlocks(
    std::vector<std::string> block_names,
    std::unordered_map<std::string, std::unordered_set<std::string>>
        block_groups)
    : num_blocks_(block_names.size()),
      block_names_(std::move(block_names)),
      block_groups_(std::move(block_groups)) {}

template <typename T, size_t Dim>
std::vector<std::array<T, Dim>> ExpandOverBlocks<T, Dim>::operator()(
    const T& value) const {
  return {num_blocks_, make_array<Dim>(value)};
}

template <typename T, size_t Dim>
std::vector<std::array<T, Dim>> ExpandOverBlocks<T, Dim>::operator()(
    const std::array<T, Dim>& value) const {
  return {num_blocks_, value};
}

template <typename T, size_t Dim>
std::vector<std::array<T, Dim>> ExpandOverBlocks<T, Dim>::operator()(
    const std::vector<std::array<T, Dim>> value) const {
  if (value.size() != num_blocks_) {
    throw std::length_error{"You supplied " + std::to_string(value.size()) +
                            " values, but the domain creator has " +
                            std::to_string(num_blocks_) + " blocks."};
  }
  return value;
}

template <typename T, size_t Dim>
std::vector<std::array<T, Dim>> ExpandOverBlocks<T, Dim>::operator()(
    const std::unordered_map<std::string, std::array<T, Dim>>& value) const {
  ASSERT(num_blocks_ == block_names_.size(),
         "Construct 'ExpandOverBlocks' with block names to use the "
         "map-over-block-names feature.");
  // Expand group names
  auto value_per_block = value;
  for (const auto& [name, block_value] : value) {
    const auto found_group = block_groups_.find(name);
    if (found_group != block_groups_.end()) {
      for (const auto& expanded_name : found_group->second) {
        if (value_per_block.count(expanded_name) == 0) {
          value_per_block[expanded_name] = block_value;
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
  std::vector<std::array<T, Dim>> result{};
  for (const auto& block_name : block_names_) {
    const auto found_value = value_per_block.find(block_name);
    if (found_value != value_per_block.end()) {
      result.push_back(found_value->second);
    } else {
      throw std::out_of_range{"Value for block '" + block_name +
                              "' is missing. Did you misspell its name?"};
    }
  }
  return result;
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM(data) BOOST_PP_TUPLE_ELEM(1, data)
#define INSTANTIATE(_, data) \
  template class ExpandOverBlocks<DTYPE(data), DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (size_t), (1, 2, 3))

#undef DIM
#undef DTYPE
#undef INSTANTIATE

}  // namespace domain
