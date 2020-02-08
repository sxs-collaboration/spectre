// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class DomainCreator.

#pragma once

#include <memory>
#include <string>

#include "DataStructures/Tensor/IndexType.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/MakeArray.hpp"

/// \cond
template <size_t>
class Block;
namespace domain {
template <typename, typename, size_t>
class CoordinateMapBase;

namespace FunctionsOfTime {
class FunctionOfTime;
}  // namespace FunctionsOfTime
}  // namespace domain
template <size_t>
class Domain;
/// \endcond

namespace domain {
/// \ingroup ComputationalDomainGroup
/// \brief Defines classes that create Domains.
namespace creators {
/// \cond
template <size_t VolumeDim>
class AlignedLattice;
class Brick;
class Cylinder;
class Disk;
class Interval;
class Rectangle;
class RotatedBricks;
class RotatedIntervals;
class RotatedRectangles;
class Shell;
class Sphere;
class FrustalCloak;
/// \endcond
}  // namespace creators
}  // namespace domain

namespace DomainCreators_detail {
template <size_t>
struct domain_creators;

template <>
struct domain_creators<1> {
  using creators = tmpl::list<domain::creators::AlignedLattice<1>,
                              domain::creators::Interval,
                              domain::creators::RotatedIntervals>;
};
template <>
struct domain_creators<2> {
  using creators =
      tmpl::list<domain::creators::AlignedLattice<2>, domain::creators::Disk,
                 domain::creators::Rectangle,
                 domain::creators::RotatedRectangles>;
};
template <>
struct domain_creators<3> {
  using creators =
      tmpl::list<domain::creators::AlignedLattice<3>, domain::creators::Brick,
                 domain::creators::Cylinder, domain::creators::RotatedBricks,
                 domain::creators::Shell, domain::creators::Sphere,
                 domain::creators::FrustalCloak>;
};
}  // namespace DomainCreators_detail

/// \ingroup ComputationalDomainGroup
/// \brief Base class for creating Domains from an option string.
template <size_t VolumeDim>
class DomainCreator {
 public:
  using creatable_classes =
      typename DomainCreators_detail::domain_creators<VolumeDim>::creators;

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

#include "Domain/Creators/AlignedLattice.hpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/Creators/Cylinder.hpp"
#include "Domain/Creators/Disk.hpp"
#include "Domain/Creators/FrustalCloak.hpp"
#include "Domain/Creators/Interval.hpp"
#include "Domain/Creators/Rectangle.hpp"
#include "Domain/Creators/RotatedBricks.hpp"
#include "Domain/Creators/RotatedIntervals.hpp"
#include "Domain/Creators/RotatedRectangles.hpp"
#include "Domain/Creators/Shell.hpp"
#include "Domain/Creators/Sphere.hpp"
