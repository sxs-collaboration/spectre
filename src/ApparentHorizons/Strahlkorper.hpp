// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "ApparentHorizons/YlmSpherepack.hpp"
#include "DataStructures/DataVector.hpp"
#include "Options/Options.hpp"
#include "Utilities/ForceInline.hpp"

namespace PUP {
class er;
}  // namespace PUP

/// \ingroup SurfacesGroup
/// \brief A star-shaped surface expanded in spherical harmonics.
template <typename Frame>
class Strahlkorper {
 public:
  struct Lmax {
    using type = size_t;
    static constexpr OptionString help = {
        "Strahlkorper is expanded in Ylms up to l=Lmax"};
  };
  struct Radius {
    using type = double;
    static constexpr OptionString help = {"Radius of spherical Strahlkorper"};
  };
  struct Center {
    using type = std::array<double, 3>;
    static constexpr OptionString help = {"Center of spherical Strahlkorper"};
  };
  using options = tmpl::list<Lmax, Radius, Center>;

  static constexpr OptionString help{
      "A star-shaped surface expressed as an expansion in spherical "
      "harmonics.\n"
      "Currently only a spherical Strahlkorper can be constructed from\n"
      "Options.  To do this, specify parameters Center, Radius, and Lmax."};

  // Pup needs default constructor
  Strahlkorper() = default;

  /// Construct a sphere of radius `radius` with a given center.
  Strahlkorper(size_t l_max, size_t m_max, double radius,
               std::array<double, 3> center) noexcept;

  /// Construct a sphere of radius `radius`, setting `m_max`=`l_max`.
  Strahlkorper(size_t l_max, double radius,
               std::array<double, 3> center) noexcept
      : Strahlkorper(l_max, l_max, radius, center) {}

  /// Construct a Strahlkorper from a DataVector containing the radius
  /// at the collocation points.
  ///
  /// \note The collocation points of the constructed Strahlkorper
  /// will not be exactly `radius_at_collocation_points`.  Instead,
  /// the constructed Strahlkorper will match the shape given by
  /// `radius_at_collocation_points` only to order (`l_max`,`m_max`).
  /// This is because the YlmSpherepack representation of the
  /// Strahlkorper has more collocation points than spectral
  /// coefficients.  Specifically, `radius_at_collocation_points` has
  /// \f$(l_{\rm max} + 1) (2 m_{\rm max} + 1)\f$ degrees of freedom,
  /// but because there are only
  /// \f$m_{\rm max}^2+(l_{\rm max}-m_{\rm max})(2m_{\rm max}+1)\f$
  /// spectral coefficients, it is not possible to choose spectral
  /// coefficients to exactly match all points in
  /// `radius_at_collocation_points`.
  Strahlkorper(size_t l_max, size_t m_max,
               const DataVector& radius_at_collocation_points,
               std::array<double, 3> center) noexcept;

  /// Prolong or restrict another surface to the given `l_max` and `m_max`.
  Strahlkorper(size_t l_max, size_t m_max,
               const Strahlkorper& another_strahlkorper) noexcept;

  /// Construct a Strahlkorper from another Strahlkorper,
  /// but explicitly specifying the coefficients.
  /// Here coefficients are in the same storage scheme
  /// as the `coefficients()` member function returns.
  Strahlkorper(DataVector coefs,
               const Strahlkorper& another_strahlkorper) noexcept;

  /// Move-construct a Strahlkorper from another Strahlkorper,
  /// explicitly specifying the coefficients.
  Strahlkorper(DataVector coefs, Strahlkorper&& another_strahlkorper) noexcept;

  // clang-tidy: no runtime references
  /// Serialization for Charm++
  void pup(PUP::er& p) noexcept;  // NOLINT

  /*!
   *  These coefficients are stored as SPHEREPACK coefficients.
   *  Suppose you represent a set of coefficients \f$F^{lm}\f$ in the expansion
   *  \f[
   *  f(\theta,\phi) =
   *  \sum_{l=0}^{l_{max}} \sum_{m=-l}^{l} F^{lm} Y^{lm}(\theta,\phi)
   *  \f]
   *  Here the \f$Y^{lm}(\theta,\phi)\f$ are the usual complex-valued scalar
   *  spherical harmonics, so \f$F^{lm}\f$ are also complex-valued.
   *  But here we assume that \f$f(\theta,\phi)\f$ is real, so therefore
   *  the \f$F^{lm}\f$ obey \f$F^{l-m} = (-1)^m (F^{lm})^\star\f$. So one
   *  does not need to store both real and imaginary parts for both positive
   *  and negative \f$m\f$, and the stored coefficients can all be real.
   *
   *  So the stored coefficients are:
   * \f{align}
   *  \text{coefficients()(l,m)} &= (-1)^m \sqrt{\frac{2}{\pi}}
   *     \Re(F^{lm}) \quad \text{for} \quad m\ge 0, \\
   *  \text{coefficients()(l,m)} &= (-1)^m \sqrt{\frac{2}{\pi}}
   *     \Im(F^{lm}) \quad \text{for} \quad m<0
   * \f}
   */
  SPECTRE_ALWAYS_INLINE const DataVector& coefficients() const noexcept {
    return strahlkorper_coefs_;
  }
  SPECTRE_ALWAYS_INLINE DataVector& coefficients() noexcept {
    return strahlkorper_coefs_;
  }

  /// Point about which the spectral basis of the Strahlkorper is expanded.
  /// The center is given in the frame in which the Strahlkorper is defined.
  /// This center must be somewhere inside the Strahlkorper, but in principle
  /// it can be anywhere.  See `physical_center()` for a different measure.
  SPECTRE_ALWAYS_INLINE const std::array<double, 3>& center() const noexcept {
    return center_;
  }

  /// Approximate physical center (determined by \f$l=1\f$ coefficients)
  /// Implementation of Eqs. (38)-(40) in \cite Hemberger2012jz
  std::array<double, 3> physical_center() const noexcept;

  /// Average radius of the surface (determined by \f$Y_{00}\f$ coefficient)
  double average_radius() const noexcept;

  /// Maximum \f$l\f$ in \f$Y_{lm}\f$ decomposition.
  SPECTRE_ALWAYS_INLINE size_t l_max() const noexcept { return l_max_; }

  /// Maximum \f$m\f$ in \f$Y_{lm}\f$ decomposition.
  SPECTRE_ALWAYS_INLINE size_t m_max() const noexcept { return m_max_; }

  /// Radius at a particular angle \f$(\theta,\phi)\f$.
  /// This is inefficient if done at multiple points many times.
  /// See YlmSpherepack for alternative ways of computing this.
  double radius(double theta, double phi) const noexcept;

  /// Determine if a point `x` is contained inside the surface.
  /// The point must be given in Cartesian coordinates in the frame in
  /// which the Strahlkorper is defined.
  /// This is inefficient if done at multiple points many times.
  bool point_is_contained(const std::array<double, 3>& x) const noexcept;

  SPECTRE_ALWAYS_INLINE const YlmSpherepack& ylm_spherepack() const noexcept {
    return ylm_;
  }

 private:
  size_t l_max_{2}, m_max_{2};
  YlmSpherepack ylm_{2, 2};
  std::array<double, 3> center_{{0.0, 0.0, 0.0}};
  DataVector strahlkorper_coefs_ = DataVector(ylm_.spectral_size(), 0.0);
};

namespace OptionTags {
/// \ingroup OptionTagsGroup
/// \ingroup SurfacesGroup
/// The input file tag for a Strahlkorper.
template <typename Frame>
struct Strahlkorper {
  using type = ::Strahlkorper<Frame>;
  static constexpr OptionString help{"A star-shaped surface"};
};
} // namespace OptionTags

template <typename Frame>
bool operator==(const Strahlkorper<Frame>& lhs,
                const Strahlkorper<Frame>& rhs) noexcept {
  return lhs.l_max() == rhs.l_max() and lhs.m_max() == rhs.m_max() and
         lhs.center() == rhs.center() and
         lhs.coefficients() == rhs.coefficients();
}

template <typename Frame>
bool operator!=(const Strahlkorper<Frame>& lhs,
                const Strahlkorper<Frame>& rhs) noexcept {
  return not(lhs == rhs);
}
