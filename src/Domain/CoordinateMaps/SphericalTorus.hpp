// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "Utilities/TypeTraits/RemoveReferenceWrapper.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

namespace domain::CoordinateMaps {
/*!
 * \ingroup CoordinateMapsGroup
 *
 * \brief Torus made by removing two polar cones from a spherical shell
 *
 * Maps source coordinates \f$(\xi, \eta, \zeta)\f$ to
 * \f{align}
 * \vec{x}(\xi, \eta, \zeta) =
 * \begin{bmatrix}
 *  r \sin\theta\cos\phi \\
 *  r \sin\theta\sin\phi \\
 *  r \cos\theta
 * \end{bmatrix}
 * \f}
 *
 * where
 * \f{align}
 *  r & = r_\mathrm{min}\frac{1-\xi}{2} + r_\mathrm{max}\frac{1+\xi}{2}, \\
 *  \theta & = \pi/2 - (\pi/2 - \theta_\mathrm{min}) \eta, \\
 *  \phi   & = f_\mathrm{torus} \pi \zeta.
 * \f}
 *
 *  - $r_\mathrm{min}$ and $r_\mathrm{max}$ are inner and outer radius of torus.
 *  - $\theta_\mathrm{min}\in(0,\pi/2)$ is the minimum polar angle (measured
 *    from +z axis) of torus, which is equal to the half of the apex angle of
 *    the removed polar cones.
 *  - $f_\mathrm{torus}\in(0, 1)$ is azimuthal fraction that the torus covers.
 *
 * \warning Internal namings of code variables in `SphericalTorus.cpp` uses
 * a different convention of angular variables for spherical coordinates.
 * Therein `theta` denotes the azimuthal angle, and `phi` denotes the elevation
 * angle measured from equator, which is equal to (\f$\pi/2\f$ - polar angle)
 * with the polar angle being measured from +z axis.
 *
 */
class SphericalTorus {
 public:
  static constexpr size_t dim = 3;

  struct RadialRange {
    using type = std::array<double, 2>;
    static constexpr Options::String help =
        "Radial extent of the torus, "
        "[min_radius, max_radius] ";
  };

  struct MinPolarAngle {
    using type = double;
    static constexpr Options::String help =
        "Half of the apex angle of excised polar cones. "
        "Polar angle (measured from +z axis) of torus has range "
        "[MinPolarAngle, pi - MinPolarAngle]";
    static type lower_bound() { return 0.0; }
    static type upper_bound() { return 0.5 * M_PI; }
  };

  struct FractionOfTorus {
    using type = double;
    static constexpr Options::String help =
        "Fraction of (azimuthal) orbit covered. Azimuthal angle has range "
        "[- pi * FractionOfTorus, pi * FractionOfTorus].";
    static type lower_bound() { return 0.0; }
    static type upper_bound() { return 1.0; }
  };

  static constexpr Options::String help =
      "Torus made by removing polar cones from a spherical shell";

  using options = tmpl::list<RadialRange, MinPolarAngle, FractionOfTorus>;

  SphericalTorus(const std::array<double, 2>& radial_range,
                 const double min_polar_angle, const double fraction_of_torus,
                 const Options::Context& context = {});

  SphericalTorus(double r_min, double r_max, double min_polar_angle,
                 double fraction_of_torus = 1.0,
                 const Options::Context& context = {});

  SphericalTorus() = default;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, 3> operator()(
      const std::array<T, 3>& source_coords) const;

  std::optional<std::array<double, 3>> inverse(
      const std::array<double, 3>& target_coords) const;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> jacobian(
      const std::array<T, 3>& source_coords) const;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> inv_jacobian(
      const std::array<T, 3>& source_coords) const;

  template <typename T>
  tnsr::Ijj<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> hessian(
      const std::array<T, 3>& source_coords) const;

  template <typename T>
  tnsr::Ijk<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>
  derivative_of_inv_jacobian(const std::array<T, 3>& source_coords) const;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

  bool is_identity() const { return false; }

 private:
  template <typename T>
  tt::remove_cvref_wrap_t<T> radius(const T& x) const;

  template <typename T>
  void radius(const gsl::not_null<tt::remove_cvref_wrap_t<T>*> r,
              const T& x) const;

  template <typename T>
  tt::remove_cvref_wrap_t<T> radius_inverse(const T& r) const;

  friend bool operator==(const SphericalTorus& lhs, const SphericalTorus& rhs);

  double r_min_ = std::numeric_limits<double>::signaling_NaN();
  double r_max_ = std::numeric_limits<double>::signaling_NaN();
  double pi_over_2_minus_theta_min_ =
      std::numeric_limits<double>::signaling_NaN();
  double fraction_of_torus_ = std::numeric_limits<double>::signaling_NaN();
};

bool operator!=(const SphericalTorus& lhs, const SphericalTorus& rhs);
}  // namespace domain::CoordinateMaps
