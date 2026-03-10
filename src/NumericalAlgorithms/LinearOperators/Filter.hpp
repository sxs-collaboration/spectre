// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>
#include <string>
#include <unordered_set>

#include "Utilities/Serialization/CharmPupable.hpp"

namespace Filters {
/*!
 * \brief Base class for filters.
 */
class Filter : public PUP::able {
 public:
  Filter() = default;
  Filter(const Filter&) = default;
  Filter(Filter&&) = default;
  Filter& operator=(const Filter&) = default;
  Filter& operator=(Filter&&) = default;
  ~Filter() override = default;

  WRAPPED_PUPable_abstract(Filter);  // NOLINT
  explicit Filter(CkMigrateMessage* m) : PUP::able(m) {}

  virtual std::optional<std::unordered_set<std::string>> blocks_to_filter()
      const = 0;
};
}  // namespace Filters
