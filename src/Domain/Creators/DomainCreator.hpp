// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class DomainCreator.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"

/// \cond
template <size_t>
class Domain;
/// \endcond

namespace domain {
/// \ingroup ComputationalDomainGroup
/// \brief Defines classes that create Domains.
namespace creators {}
}  // namespace domain

/// \ingroup ComputationalDomainGroup
/// \brief Base class for creating Domains from an option string.
template <size_t VolumeDim>
class DomainCreator {
 public:
  DomainCreator() = default;
  DomainCreator(const DomainCreator<VolumeDim>&) = delete;
  DomainCreator(DomainCreator<VolumeDim>&&) = default;
  DomainCreator<VolumeDim>& operator=(const DomainCreator<VolumeDim>&) = delete;
  DomainCreator<VolumeDim>& operator=(DomainCreator<VolumeDim>&&) = default;
  virtual ~DomainCreator() = default;

  virtual Domain<VolumeDim> create_domain() const = 0;

  /// A human-readable name for every block, or empty if the domain creator
  /// doesn't support block names (yet).
  virtual std::vector<std::string> block_names() const { return {}; }

  /// Labels to refer to groups of blocks. The groups can overlap, and they
  /// don't have to cover all blocks in the domain. The groups can be used to
  /// refer to multiple blocks at once when specifying input-file options.
  virtual std::unordered_map<std::string, std::unordered_set<std::string>>
  block_groups() const {
    return {};
  }

  /// Obtain the initial grid extents of the Element%s in each block.
  virtual std::vector<std::array<size_t, VolumeDim>> initial_extents()
      const = 0;

  /// Obtain the initial refinement levels of the blocks.
  virtual std::vector<std::array<size_t, VolumeDim>> initial_refinement_levels()
      const = 0;

  /// Retrieve the functions of time used for moving meshes.
  virtual auto functions_of_time() const -> std::unordered_map<
      std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>> {
    return {};
  }
};
