// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class ExcisionSphere.

#pragma once

#include <array>
#include <cstddef>
#include <iosfwd>
#include <limits>

#include "Utilities/MakeArray.hpp"

/// \cond
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
  ExcisionSphere(double radius, std::array<double, VolumeDim> center) noexcept;

  /// Default constructor needed for Charm++ serialization.
  ExcisionSphere() noexcept = default;
  ~ExcisionSphere() noexcept = default;
  ExcisionSphere(const ExcisionSphere<VolumeDim>& /*rhs*/) noexcept = default;
  ExcisionSphere(ExcisionSphere<VolumeDim>&& /*rhs*/) noexcept = default;
  ExcisionSphere<VolumeDim>& operator=(
      const ExcisionSphere<VolumeDim>& /*rhs*/) noexcept = default;
  ExcisionSphere<VolumeDim>& operator=(
      ExcisionSphere<VolumeDim>&& /*rhs*/) noexcept = default;

  /// The radius of the ExcisionSphere.
  double radius() const noexcept { return radius_; }

  /// The coodinate center of the ExcisionSphere.
  const std::array<double, VolumeDim>& center() const noexcept {
    return center_;
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept;

 private:
  double radius_{std::numeric_limits<double>::signaling_NaN()};
  std::array<double, VolumeDim> center_{
      make_array<VolumeDim>(std::numeric_limits<double>::signaling_NaN())};
};

template <size_t VolumeDim>
std::ostream& operator<<(
    std::ostream& os,
    const ExcisionSphere<VolumeDim>& excision_sphere) noexcept;

template <size_t VolumeDim>
bool operator==(const ExcisionSphere<VolumeDim>& lhs,
                const ExcisionSphere<VolumeDim>& rhs) noexcept;

template <size_t VolumeDim>
bool operator!=(const ExcisionSphere<VolumeDim>& lhs,
                const ExcisionSphere<VolumeDim>& rhs) noexcept;
