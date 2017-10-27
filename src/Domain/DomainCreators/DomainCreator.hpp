// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class DomainCreator.

#pragma once

#include <memory>
#include <string>

#include "DataStructures/Tensor/IndexType.hpp"
#include "Options/Factory.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/MakeArray.hpp"

template <size_t, typename>
class Block;

template <typename, typename, size_t>
class CoordinateMapBase;

template <size_t, typename>
class Domain;

/// Defines classes that create Domains.
namespace DomainCreators {
class Brick;
// class Disk;
class Interval;
class Rectangle;
// class RotatedBricks;
// class RotatedIntervals;
// class RotatedRectangles;
// class RubiksCubeWithHole;
// class Shell;
// class Sphere;
// class UnevenBricks;
}  // namespace DomainCreators

namespace DomainCreators_detail {
template <size_t>
struct domain_creators;

template <>
struct domain_creators<1> {
  using type = typelist<DomainCreators::Interval
                        //, DomainCreators::RotatedIntervals
                        >;
};
template <>
struct domain_creators<2> {
  using type =
      typelist<DomainCreators::Rectangle  //, DomainCreators::Disk,
                                          // DomainCreators::RotatedRectangles
               >;
};
template <>
struct domain_creators<3> {
  using type = typelist<
      DomainCreators::Brick  //, DomainCreators::RubiksCubeWithHole,
                             //                DomainCreators::Shell,
                             //                DomainCreators::Sphere,
                             //                DomainCreators::RotatedBricks,
                             //                DomainCreators::UnevenBricks
      >;
};
}  // namespace DomainCreators_detail

/// Base class for creating Domains from an option string.
template <size_t VolumeDim, typename TargetFrame>
class DomainCreator : public Factory<DomainCreator<VolumeDim, TargetFrame>> {
 public:
  using creatable_classes =
      typename DomainCreators_detail::domain_creators<VolumeDim>::type;

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
  virtual std::array<size_t, VolumeDim> initial_extents(
      size_t block_index) const = 0;

  /// Obtain the initial refinement levels of the block with the given index.
  virtual std::array<size_t, VolumeDim> initial_refinement_levels(
      size_t block_index) const = 0;
};

#include "Domain/DomainCreators/Brick.hpp"
// #include "Domain/DomainCreators/Disk.hpp"
#include "Domain/DomainCreators/Interval.hpp"
#include "Domain/DomainCreators/Rectangle.hpp"
// #include "Domain/DomainCreators/RotatedBricks.hpp"
// #include "Domain/DomainCreators/RotatedIntervals.hpp"
// #include "Domain/DomainCreators/RotatedRectangles.hpp"
// #include "Domain/DomainCreators/RubiksCubeWithHole.hpp"
// #include "Domain/DomainCreators/Shell.hpp"
// #include "Domain/DomainCreators/Sphere.hpp"
// #include "Domain/DomainCreators/UnevenBricks.hpp"
