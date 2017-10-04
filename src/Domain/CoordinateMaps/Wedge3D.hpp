// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines the class Wedge3D.

#pragma once

#include <array>
#include <limits>
#include <memory>
#include <pup.h>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Direction.hpp"

namespace CoordinateMaps {

/*!
 * \ingroup CoordinateMaps
 *
 * \brief Three dimensional map from the cube to a wedge.
 *
 * \details The mapping that goes from a reference cube to a three-dimensional
 *  wedge centered on a coordinate axis covering a volume between an inner
 *  surface and outer surface. One of the surfaces must be spherical, but the
 *  curvature of the other surface can be anything between flat (a sphericity of
 *  0) and spherical (a sphericity of 1).
 *
 *  The first two logical coordinates correspond to the two angular coordinates,
 *  and the third to the radial coordinate
 */
class Wedge3D {
 public:
  static constexpr size_t dim = 3;

  /*!
   * Constructs a 3D wedge.
   * \param radius_of_spherical_surface Radius of the spherical surface
   * \param radius_of_other_surface Distance from the origin to one of the
   * corners which lie on the other surface, which may be anything between flat
   * and spherical.
   * \param direction_of_wedge The axis on which the
   * wedge is centred.
   * \param sphericity_of_other_surface Value between 0 and 1 which determines
   * whether the other surface is flat (value of 0), spherical (value of 1) or
   * somewhere in between
   */
  Wedge3D(double radius_of_other_surface, double radius_of_spherical_surface,
          Direction<3> direction_of_wedge,
          double sphericity_of_other_surface) noexcept;

  Wedge3D() = default;
  ~Wedge3D() = default;
  Wedge3D(Wedge3D&&) = default;
  Wedge3D(const Wedge3D&) = default;
  Wedge3D& operator=(const Wedge3D&) = default;
  Wedge3D& operator=(Wedge3D&&) = default;

  template <typename T>
  std::array<std::decay_t<tt::remove_reference_wrapper_t<T>>, 3> operator()(
      const std::array<T, 3>& x) const noexcept;

  // Currently unimplemented, hence the noreturn attribute
  template <typename T>
  [[noreturn]] std::array<std::decay_t<tt::remove_reference_wrapper_t<T>>, 3>
  inverse(const std::array<T, 3>& x) const noexcept;

  template <typename T>
  Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
         tmpl::integral_list<std::int32_t, 2, 1>,
         index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                    SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
  jacobian(const std::array<T, 3>& x) const noexcept;

  template <typename T>
  Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
         tmpl::integral_list<std::int32_t, 2, 1>,
         index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                    SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
  inv_jacobian(const std::array<T, 3>& xi) const noexcept;

  // clang-tidy: google runtime references
  void pup(PUP::er& p);  // NOLINT

 private:
  friend bool operator==(const Wedge3D& lhs, const Wedge3D& rhs) noexcept;

  double radius_of_other_surface_{std::numeric_limits<double>::signaling_NaN()};
  double radius_of_spherical_surface_{
      std::numeric_limits<double>::signaling_NaN()};
  Direction<3> direction_of_wedge_{};
  double sphericity_of_other_surface_{
      std::numeric_limits<double>::signaling_NaN()};
};
bool operator!=(const Wedge3D& lhs, const Wedge3D& rhs) noexcept;
}  // namespace CoordinateMaps
