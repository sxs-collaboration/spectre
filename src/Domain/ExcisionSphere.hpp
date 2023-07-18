// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class ExcisionSphere.

#pragma once

#include <cstddef>
#include <iosfwd>
#include <limits>
#include <memory>
#include <optional>
#include <unordered_map>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"

/// \cond
namespace Frame {
struct Grid;
struct Inertial;
}  // namespace Frame
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

/// \ingroup ComputationalDomainGroup
/// The excision sphere information of a computational domain.
/// The excision sphere is assumed to be a coordinate sphere in the
/// grid frame.
///
/// \tparam VolumeDim the volume dimension.
template <size_t VolumeDim>
class ExcisionSphere {
 public:
  /// Constructor
  ///
  /// \param radius the radius of the excision sphere in the
  /// computational domain.
  /// \param center the coordinate center of the excision sphere
  /// in the computational domain.
  /// \param abutting_directions the set of blocks that touch the excision
  /// sphere, along with the direction in which they touch it.
  ExcisionSphere(
      double radius, tnsr::I<double, VolumeDim, Frame::Grid> center,
      std::unordered_map<size_t, Direction<VolumeDim>> abutting_directions);

  /// Default constructor needed for Charm++ serialization.
  ExcisionSphere() = default;
  ~ExcisionSphere() = default;
  ExcisionSphere(const ExcisionSphere<VolumeDim>& /*rhs*/);
  ExcisionSphere(ExcisionSphere<VolumeDim>&& /*rhs*/) = default;
  ExcisionSphere<VolumeDim>& operator=(
      const ExcisionSphere<VolumeDim>& /*rhs*/);
  ExcisionSphere<VolumeDim>& operator=(ExcisionSphere<VolumeDim>&& /*rhs*/) =
      default;

  /// \brief Add time dependent coordinate maps to the ExcisionSphere
  ///
  /// \note There is no actual mesh that is inside an excision sphere, but this
  /// region moves along with the domain just the same. Meaning that we
  /// should be able to take a point inside the excision sphere and map it to
  /// the Inertial frame.
  void inject_time_dependent_maps(
      std::unique_ptr<
          domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, VolumeDim>>
          moving_mesh_grid_to_inertial_map);

  /// \brief The map going from the last time independent frame to the frame in
  /// which the equations are solved. Only used when `is_time_dependent()` is
  /// true.
  const domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, VolumeDim>&
  moving_mesh_grid_to_inertial_map() const;

  /// \brief Return whether or not time dependent maps have been injected
  bool is_time_dependent() const { return grid_to_inertial_map_ != nullptr; }

  /// The radius of the ExcisionSphere.
  double radius() const { return radius_; }

  /// The coordinate center of the ExcisionSphere.
  const tnsr::I<double, VolumeDim, Frame::Grid>& center() const {
    return center_;
  }

  /// The set of blocks that touch the excision sphere, along with the direction
  /// in which they touch it
  const std::unordered_map<size_t, Direction<VolumeDim>>& abutting_directions()
      const {
    return abutting_directions_;
  }
  /// Checks whether an element abuts the excision sphere. If it does, returns
  /// the corresponding direction. Else, `nullopt` is returned.
  std::optional<Direction<VolumeDim>> abutting_direction(
      const ElementId<VolumeDim>& element_id) const;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

 private:
  template <size_t LocalVolumeDim>
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator==(const ExcisionSphere<LocalVolumeDim>& lhs,
                         const ExcisionSphere<LocalVolumeDim>& rhs);

  double radius_{std::numeric_limits<double>::signaling_NaN()};
  tnsr::I<double, VolumeDim, Frame::Grid> center_{
      std::numeric_limits<double>::signaling_NaN()};
  std::unordered_map<size_t, Direction<VolumeDim>> abutting_directions_{};
  std::unique_ptr<
      domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, VolumeDim>>
      grid_to_inertial_map_{};
};

template <size_t VolumeDim>
std::ostream& operator<<(std::ostream& os,
                         const ExcisionSphere<VolumeDim>& excision_sphere);

template <size_t VolumeDim>
bool operator!=(const ExcisionSphere<VolumeDim>& lhs,
                const ExcisionSphere<VolumeDim>& rhs);
