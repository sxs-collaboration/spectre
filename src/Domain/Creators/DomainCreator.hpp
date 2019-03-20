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
template <size_t, typename>
class Block;
namespace domain {
template <typename, typename, size_t>
class CoordinateMapBase;
}  // namespace domain
template <size_t, typename>
class Domain;
/// \endcond

namespace domain {
/// Defines classes that create Domains.
namespace creators {
/// \cond
template <size_t VolumeDim, typename TargetFrame>
class AlignedLattice;
template <typename TargetFrame>
class Brick;
template <typename TargetFrame>
class Cylinder;
template <typename TargetFrame>
class Disk;
template <typename TargetFrame>
class Interval;
template <typename TargetFrame>
class Rectangle;
template <typename TargetFrame>
class RotatedBricks;
template <typename TargetFrame>
class RotatedIntervals;
template <typename TargetFrame>
class RotatedRectangles;
template <typename TargetFrame>
class Shell;
template <typename TargetFrame>
class Sphere;
template <typename TargetFrame>
class FrustalCloak;
/// \endcond
}  // namespace creators
}  // namespace domain

namespace DomainCreators_detail {
template <size_t>
struct domain_creators;

template <>
struct domain_creators<1> {
  template <typename Frame>
  using creators = tmpl::list<domain::creators::AlignedLattice<1, Frame>,
                              domain::creators::Interval<Frame>,
                              domain::creators::RotatedIntervals<Frame>>;
};
template <>
struct domain_creators<2> {
  template <typename Frame>
  using creators = tmpl::list<domain::creators::AlignedLattice<2, Frame>,
                              domain::creators::Disk<Frame>,
                              domain::creators::Rectangle<Frame>,
                              domain::creators::RotatedRectangles<Frame>>;
};
template <>
struct domain_creators<3> {
  template <typename Frame>
  using creators = tmpl::list<
      domain::creators::AlignedLattice<3, Frame>,
      domain::creators::Brick<Frame>, domain::creators::Cylinder<Frame>,
      domain::creators::RotatedBricks<Frame>, domain::creators::Shell<Frame>,
      domain::creators::Sphere<Frame>, domain::creators::FrustalCloak<Frame>>;
};
}  // namespace DomainCreators_detail

/// Base class for creating Domains from an option string.
template <size_t VolumeDim, typename TargetFrame>
class DomainCreator {
 public:
  using creatable_classes = typename DomainCreators_detail::domain_creators<
      VolumeDim>::template creators<TargetFrame>;

  DomainCreator() = default;
  DomainCreator(const DomainCreator<VolumeDim, TargetFrame>&) = delete;
  DomainCreator(DomainCreator<VolumeDim, TargetFrame>&&) noexcept = default;
  DomainCreator<VolumeDim, TargetFrame>& operator=(
      const DomainCreator<VolumeDim, TargetFrame>&) = delete;
  DomainCreator<VolumeDim, TargetFrame>& operator=(
      DomainCreator<VolumeDim, TargetFrame>&&) noexcept = default;
  virtual ~DomainCreator() = default;

  virtual Domain<VolumeDim, TargetFrame> create_domain() const = 0;

  /// Obtain the initial grid extents of the block with the given index.
  virtual std::vector<std::array<size_t, VolumeDim>> initial_extents() const
      noexcept = 0;

  /// Obtain the initial refinement levels of the blocks.
  virtual std::vector<std::array<size_t, VolumeDim>> initial_refinement_levels()
      const noexcept = 0;
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
