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
namespace domain {
template <size_t, typename>
class Block;
template <typename, typename, size_t>
class CoordinateMapBase;
template <size_t, typename>
class Domain;
}  // namespace domain
/// \endcond

namespace domain {
/// Defines classes that create Domains.
namespace creators {
/// \cond
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
/// \endcond
}  // namespace creators

namespace DomainCreators_detail {
template <size_t>
struct domain_creators;

template <>
struct domain_creators<1> {
  template <typename Frame>
  using creators = tmpl::list<domain::creators::Interval<Frame>,
                              domain::creators::RotatedIntervals<Frame>>;
};
template <>
struct domain_creators<2> {
  template <typename Frame>
  using creators = tmpl::list<domain::creators::Disk<Frame>,
                              domain::creators::Rectangle<Frame>,
                              domain::creators::RotatedRectangles<Frame>>;
};
template <>
struct domain_creators<3> {
  template <typename Frame>
  using creators = tmpl::list<
      domain::creators::Brick<Frame>, domain::creators::Cylinder<Frame>,
      domain::creators::RotatedBricks<Frame>, domain::creators::Shell<Frame>,
      domain::creators::Sphere<Frame>>;
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
}  // namespace domain

#include "Domain/DomainCreators/Brick.hpp"
#include "Domain/DomainCreators/Cylinder.hpp"
#include "Domain/DomainCreators/Disk.hpp"
#include "Domain/DomainCreators/Interval.hpp"
#include "Domain/DomainCreators/Rectangle.hpp"
#include "Domain/DomainCreators/RotatedBricks.hpp"
#include "Domain/DomainCreators/RotatedIntervals.hpp"
#include "Domain/DomainCreators/RotatedRectangles.hpp"
#include "Domain/DomainCreators/Shell.hpp"
#include "Domain/DomainCreators/Sphere.hpp"
