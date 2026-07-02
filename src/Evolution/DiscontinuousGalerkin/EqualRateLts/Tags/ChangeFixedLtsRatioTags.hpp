// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
#include "Utilities/TMPL.hpp"

/// Tags used by the ChangeFixedLtsRatio action.
namespace evolution::dg::Tags::ChangeFixedLtsRatio {
/// Number of expected messages to the ChangeFixedLtsRatio action.
struct NumberOfExpectedMessages : db::SimpleTag {
  using type = std::map<std::pair<int64_t, uint64_t>, size_t>;
};

/// New step size requests for the ChangeFixedLtsRatio action.
struct NewStepSize : db::SimpleTag {
  using type = std::map<std::pair<int64_t, uint64_t>, std::vector<double>>;
};
}  // namespace evolution::dg::Tags::ChangeFixedLtsRatio
