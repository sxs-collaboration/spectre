// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "NumericalAlgorithms/LinearOperators/Filters/None.hpp"

#include <optional>
#include <pup_stl.h>
#include <string>
#include <unordered_set>
#include <vector>

#include "Domain/Structure/BlockGroups.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/ParseError.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"

namespace Filters {
template <size_t Dim, typename TagList>
None<Dim, TagList>::None(
    const std::optional<std::vector<std::string>>& blocks_to_filter,
    const Options::Context& context) {
  if (blocks_to_filter.has_value()) {
    blocks_and_groups_to_filter_ = std::vector<std::string>{};
    std::unordered_set<std::string> seen{};
    for (const std::string& block_name : blocks_to_filter.value()) {
      if (not seen.emplace(block_name).second) {
        PARSE_ERROR(context, "Duplicate block name '"
                                 << block_name
                                 << "' found when creating a None filter.");
      }
      blocks_and_groups_to_filter_->push_back(block_name);
    }
  }
}

template <size_t Dim, typename TagList>
void None<Dim, TagList>::pup(PUP::er& p) {
  Filter<Dim, TagList>::pup(p);
  p | blocks_and_groups_to_filter_;
  p | blocks_to_filter_;
}

template <size_t Dim, typename TagList>
const std::optional<std::vector<size_t>>& None<Dim, TagList>::blocks_to_filter()
    const {
  return blocks_to_filter_;
}

template <size_t Dim, typename TagList>
void None<Dim, TagList>::set_blocks_to_filter(
    const std::vector<std::string>& all_block_names,
    const std::unordered_map<std::string, std::unordered_set<std::string>>&
        block_groups) {
  if (not blocks_and_groups_to_filter_.has_value()) {
    blocks_to_filter_ = std::nullopt;
    return;
  }
  if (all_block_names.empty()) {
    ERROR(
        "The domain chosen doesn't use block names, but the filter has "
        "specified block names to use.");
  }
  blocks_to_filter_ = domain::block_ids_from_names(
      blocks_and_groups_to_filter_.value(), all_block_names, block_groups);
}

template <size_t Dim, typename TagList>
void None<Dim, TagList>::apply_in_volume(
    const gsl::not_null<Variables<TagList>*> /*vars*/,
    const Mesh<Dim>& /*mesh*/,
    const std::optional<
        InverseJacobian<DataVector, Dim, Frame::Grid, Frame::Inertial>>&
    /*inv_jac_grid_to_inertial*/,
    const std::optional<
        Jacobian<DataVector, Dim, Frame::Grid, Frame::Inertial>>&
    /*jac_grid_to_inertial*/) const {}

template <size_t Dim, typename TagList>
void None<Dim, TagList>::apply_on_boundary(
    const gsl::not_null<Variables<TagList>*> /*vars*/,
    const Mesh<Dim - 1>& /*mesh*/,
    const std::optional<
        InverseJacobian<DataVector, Dim, Frame::Grid, Frame::Inertial>>&
    /*inv_jac_grid_to_inertial*/,
    const std::optional<
        Jacobian<DataVector, Dim, Frame::Grid, Frame::Inertial>>&
    /*jac_grid_to_inertial*/) const {}

template <size_t Dim, typename TagList>
bool operator==(const None<Dim, TagList>& lhs, const None<Dim, TagList>& rhs) {
  return lhs.blocks_and_groups_to_filter_ ==
             rhs.blocks_and_groups_to_filter_ and
         lhs.blocks_to_filter_ == rhs.blocks_to_filter_;
}

template <size_t Dim, typename TagList>
bool operator!=(const None<Dim, TagList>& lhs, const None<Dim, TagList>& rhs) {
  return not(lhs == rhs);
}

template <size_t Dim, typename TagList>
bool None<Dim, TagList>::is_equal(const Filter<Dim, TagList>& other) const {
  const auto* const other_none =
      dynamic_cast<const None<Dim, TagList>*>(&other);
  if (other_none == nullptr) {
    return false;
  }
  return *this == *other_none;
}

template <size_t Dim, typename TagList>
PUP::able::PUP_ID None<Dim, TagList>::my_PUP_ID = 0;  // NOLINT
}  // namespace Filters
