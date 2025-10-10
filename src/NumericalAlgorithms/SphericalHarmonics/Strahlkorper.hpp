// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/ModalVector.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP

namespace Options {
template <typename... AlternativeLists>
struct Alternatives;
}  // namespace Options
/// \endcond

namespace ylm {
/// \ingroup SurfacesGroup
/// \brief A star-shaped surface expanded in spherical harmonics.
template <typename Frame>
class Strahlkorper {
 public:
  struct LMax {
    using type = size_t;
    static constexpr Options::String help = {
        "Strahlkorper is expanded in Ylms up to l=LMax"};
  };
  struct Radius {
    using type = double;
    static constexpr Options::String help = {
        "Radius of spherical Strahlkorper"};
  };
  struct Center {
    using type = std::array<double, 3>;
    static constexpr Options::String help = {
        "Center of spherical Strahlkorper"};
  };
  struct H5Filename {
    using type = std::string;
    static constexpr Options::String help = {
        "H5 file containing the Strahlkorper coefficients"};
  };
  struct SubfileName {
    using type = std::string;
    static constexpr Options::String help = {
        "Subfile (without leading slash or .dat extension) within "
        "the H5 file that contains the Strahlkorper coefficients"};
  };
  struct Time {
    using type = double;
    static constexpr Options::String help = {
        "Time at which to read the Strahlkorper coefficients"};
  };
  struct TimeEpsilon {
    using type = double;
    static constexpr Options::String help = {
        "Tolerance for matching the requested time to read the Strahlkorper "
        "coefficients"};
  };
  struct CheckFrame {
    using type = bool;
    static constexpr Options::String help = {
        "Whether to check that the frame in the file matches the requested "
        "frame"};
  };
  using options =
      tmpl::list<LMax,
                 Options::Alternatives<tmpl::list<Radius, Center>,
                                       tmpl::list<H5Filename, SubfileName, Time,
                                                  TimeEpsilon, CheckFrame>>>;

  static constexpr Options::String help{
      "A star-shaped surface expressed as an expansion in spherical "
      "harmonics.\n"
      "A Strahlkorper can be constructed from Options in two ways:\n"
      "1. Specify LMax, Center, and Radius to construct a spherical "
      "Strahlkorper.\n"
      "2. Specify LMax, H5Filename, SubfileName, Time, TimeEpsilon, and "
      "CheckFrame to construct a Strahlkorper from coefficients read from an "
      "H5 file. The Strahlkorper will be prolonged or restricted to the "
      "specified LMax."};

  // Pup needs default constructor
  Strahlkorper() = default;
  Strahlkorper(const Strahlkorper&) = default;
  Strahlkorper(Strahlkorper&&) = default;
  Strahlkorper& operator=(const Strahlkorper&) = default;
  Strahlkorper& operator=(Strahlkorper&&) = default;

  /// Construct a sphere of radius `radius` with a given center.
  Strahlkorper(size_t l_max, size_t m_max, double radius,
               std::array<double, 3> center);

  /// Construct a sphere of radius `radius`, setting `m_max`=`l_max`.
  Strahlkorper(size_t l_max, double radius, std::array<double, 3> center)
      : Strahlkorper(l_max, l_max, radius, center) {}

  /// Construct from options: either from radius and center, or from file.
  /// These constructors handle both alternatives specified in the options.
  Strahlkorper(size_t l_max, const std::string& h5_filename,
               const std::string& subfile_name, double time,
               double time_epsilon, bool check_frame,
               const Options::Context& context);

  /// Construct a Strahlkorper from a DataVector containing the radius
  /// at the collocation points.
  ///
  /// \note The collocation points of the constructed Strahlkorper
  /// will not be exactly `radius_at_collocation_points`.  Instead,
  /// the constructed Strahlkorper will match the shape given by
  /// `radius_at_collocation_points` only to order (`l_max`,`m_max`).
  /// This is because the ylm::Spherepack representation of the
  /// Strahlkorper has more collocation points than spectral
  /// coefficients.  Specifically, `radius_at_collocation_points` has
  /// \f$(l_{\rm max} + 1) (2 m_{\rm max} + 1)\f$ degrees of freedom,
  /// but because there are only
  /// \f$m_{\rm max}^2+(l_{\rm max}-m_{\rm max})(2m_{\rm max}+1)\f$
  /// spectral coefficients, it is not possible to choose spectral
  /// coefficients to exactly match all points in
  /// `radius_at_collocation_points`.
  explicit Strahlkorper(size_t l_max, size_t m_max,
                        const DataVector& radius_at_collocation_points,
                        std::array<double, 3> center);

  /// \brief Construct a Strahlkorper from a `ModalVector` containing the
  /// spectral coefficients
  ///
  /// \details The spectral coefficients should be in the form defined by
  /// `ylm::Strahlkorper::coefficients() const`.
  explicit Strahlkorper(size_t l_max, size_t m_max,
                        const ModalVector& spectral_coefficients,
                        std::array<double, 3> center);

  /// Copies a Strahlkorper from another frame into this Strahlkorper.
  ///
  /// \note The `OtherFrame` is ignored.
  template <typename OtherFrame>
  explicit Strahlkorper(const Strahlkorper<OtherFrame>& another_strahlkorper)
      : l_max_(another_strahlkorper.l_max()),
        m_max_(another_strahlkorper.m_max()),
        ylm_(l_max_, m_max_),
        center_(another_strahlkorper.expansion_center()),
        strahlkorper_coefs_(another_strahlkorper.coefficients()) {}

  /// Prolong or restrict another surface to the given `l_max` and `m_max`.
  ///
  /// \note The `OtherFrame` is ignored.
  template <typename OtherFrame>
  Strahlkorper(size_t l_max, size_t m_max,
               const Strahlkorper<OtherFrame>& another_strahlkorper)
      : l_max_(l_max),
        m_max_(m_max),
        ylm_(l_max, m_max),
        center_(another_strahlkorper.expansion_center()),
        strahlkorper_coefs_(
            another_strahlkorper.ylm_spherepack().prolong_or_restrict(
                another_strahlkorper.coefficients(), ylm_)) {}

  /// Construct a Strahlkorper from another Strahlkorper,
  /// but explicitly specifying the coefficients.
  /// Here coefficients are in the same storage scheme
  /// as the `coefficients()` member function returns.
  ///
  /// \note The `OtherFrame` is ignored.
  template <typename OtherFrame>
  Strahlkorper(DataVector coefs,
               const Strahlkorper<OtherFrame>& another_strahlkorper)
      : l_max_(another_strahlkorper.l_max()),
        m_max_(another_strahlkorper.m_max()),
        ylm_(another_strahlkorper.ylm_spherepack()),
        center_(another_strahlkorper.expansion_center()),
        strahlkorper_coefs_(std::move(coefs)) {
    ASSERT(
        strahlkorper_coefs_.size() ==
            another_strahlkorper.ylm_spherepack().spectral_size(),
        "Bad size " << strahlkorper_coefs_.size() << ", expected "
                    << another_strahlkorper.ylm_spherepack().spectral_size());
  }

  /// Serialization for Charm++
  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

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
  SPECTRE_ALWAYS_INLINE const DataVector& coefficients() const {
    return strahlkorper_coefs_;
  }
  SPECTRE_ALWAYS_INLINE DataVector& coefficients() {
    return strahlkorper_coefs_;
  }

  /// Point about which the spectral basis of the Strahlkorper is expanded.
  /// The center is given in the frame in which the Strahlkorper is defined.
  /// This center must be somewhere inside the Strahlkorper, but in principle
  /// it can be anywhere.  See `physical_center()` for a different measure.
  SPECTRE_ALWAYS_INLINE const std::array<double, 3>& expansion_center() const {
    return center_;
  }

  /// Approximate physical center (determined by \f$l=1\f$ coefficients)
  /// Implementation of Eqs. (38)-(40) in \cite Hemberger2012jz
  std::array<double, 3> physical_center() const;

  /// Average radius of the surface (determined by \f$Y_{00}\f$ coefficient)
  double average_radius() const;

  /// Maximum \f$l\f$ in \f$Y_{lm}\f$ decomposition.
  SPECTRE_ALWAYS_INLINE size_t l_max() const { return l_max_; }

  /// Maximum \f$m\f$ in \f$Y_{lm}\f$ decomposition.
  SPECTRE_ALWAYS_INLINE size_t m_max() const { return m_max_; }

  /// Radius at a particular angle \f$(\theta,\phi)\f$.
  /// This is inefficient if done at multiple points many times.
  /// See ylm::Spherepack for alternative ways of computing this.
  double radius(double theta, double phi) const;

  /// Determine if a point `x` is contained inside the surface.
  /// The point must be given in Cartesian coordinates in the frame in
  /// which the Strahlkorper is defined.
  /// This is inefficient if done at multiple points many times.
  bool point_is_contained(const std::array<double, 3>& x) const;

  SPECTRE_ALWAYS_INLINE const ylm::Spherepack& ylm_spherepack() const {
    return ylm_;
  }

 private:
  size_t l_max_{2}, m_max_{2};
  ylm::Spherepack ylm_{2, 2};
  std::array<double, 3> center_{{0.0, 0.0, 0.0}};
  DataVector strahlkorper_coefs_ = DataVector(ylm_.spectral_size(), 0.0);
};

namespace OptionTags {
/// \ingroup OptionTagsGroup
/// \ingroup SurfacesGroup
/// The input file tag for a Strahlkorper.
template <typename Frame>
struct Strahlkorper {
  using type = ylm::Strahlkorper<Frame>;
  static constexpr Options::String help{"A star-shaped surface"};
};
}  // namespace OptionTags

template <typename Frame>
bool operator==(const Strahlkorper<Frame>& lhs,
                const Strahlkorper<Frame>& rhs) {
  return lhs.l_max() == rhs.l_max() and lhs.m_max() == rhs.m_max() and
         lhs.expansion_center() == rhs.expansion_center() and
         lhs.coefficients() == rhs.coefficients();
}

template <typename Frame>
bool operator!=(const Strahlkorper<Frame>& lhs,
                const Strahlkorper<Frame>& rhs) {
  return not(lhs == rhs);
}
}  // namespace ylm
