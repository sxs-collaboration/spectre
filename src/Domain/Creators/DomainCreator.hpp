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
  DomainCreator(DomainCreator<VolumeDim>&&) noexcept = default;
  DomainCreator<VolumeDim>& operator=(const DomainCreator<VolumeDim>&) = delete;
  DomainCreator<VolumeDim>& operator=(DomainCreator<VolumeDim>&&) noexcept =
      default;
  virtual ~DomainCreator() = default;

  virtual Domain<VolumeDim> create_domain() const = 0;

  /// Obtain the initial grid extents of the Element%s in each block.
  virtual std::vector<std::array<size_t, VolumeDim>> initial_extents() const
      noexcept = 0;

  /// Obtain the initial refinement levels of the blocks.
  virtual std::vector<std::array<size_t, VolumeDim>> initial_refinement_levels()
      const noexcept = 0;

  /// Retrieve the functions of time used for moving meshes.
  virtual auto functions_of_time() const noexcept -> std::unordered_map<
      std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>> {
    return {};
  }
};
