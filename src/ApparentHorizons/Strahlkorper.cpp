// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ApparentHorizons/Strahlkorper.hpp"

#include <cmath>
#include <ostream>
#include <pup.h>
#include <utility>

#include "ApparentHorizons/SpherepackIterator.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Utilities/StdArrayHelpers.hpp"
/// \cond
namespace Frame {
struct Inertial;
}  // namespace Frame
/// \endcond

template <typename Frame>
Strahlkorper<Frame>::Strahlkorper(const size_t l_max, const size_t m_max,
                                  const double radius,
                                  std::array<double, 3> center) noexcept
    : l_max_(l_max),
      m_max_(m_max),
      ylm_(l_max, m_max),
      // clang-tidy: do not std::move trivially constructable types
      center_(std::move(center)),  // NOLINT
      strahlkorper_coefs_(ylm_.spectral_size(), 0.0) {
  ylm_.add_constant(&strahlkorper_coefs_, radius);
}

template <typename Frame>
Strahlkorper<Frame>::Strahlkorper(
    const size_t l_max, const size_t m_max,
    const DataVector& radius_at_collocation_points,
    std::array<double, 3> center) noexcept
    : l_max_(l_max),
      m_max_(m_max),
      ylm_(l_max, m_max),
      // clang-tidy: do not std::move trivially constructable types
      center_(std::move(center)),  // NOLINT
      strahlkorper_coefs_(ylm_.phys_to_spec(radius_at_collocation_points)) {
  ASSERT(radius_at_collocation_points.size() == ylm_.physical_size(),
         "Bad size " << radius_at_collocation_points.size() << ", expected "
                     << ylm_.physical_size());
}

template <typename Frame>
Strahlkorper<Frame>::Strahlkorper(
    const size_t l_max, const size_t m_max,
    const Strahlkorper& another_strahlkorper) noexcept
    : l_max_(l_max),
      m_max_(m_max),
      ylm_(l_max, m_max),
      center_(another_strahlkorper.center_),
      strahlkorper_coefs_(another_strahlkorper.ylm_.prolong_or_restrict(
          another_strahlkorper.strahlkorper_coefs_, ylm_)) {}

template <typename Frame>
Strahlkorper<Frame>::Strahlkorper(
    DataVector coefs, const Strahlkorper& another_strahlkorper) noexcept
    : l_max_(another_strahlkorper.l_max_),
      m_max_(another_strahlkorper.m_max_),
      ylm_(another_strahlkorper.ylm_),
      center_(another_strahlkorper.center_),
      strahlkorper_coefs_(std::move(coefs)) {
  ASSERT(
      strahlkorper_coefs_.size() == another_strahlkorper.ylm_.spectral_size(),
      "Bad size " << strahlkorper_coefs_.size() << ", expected "
                  << another_strahlkorper.ylm_.spectral_size());
}

template <typename Frame>
Strahlkorper<Frame>::Strahlkorper(DataVector coefs,
                                  Strahlkorper&& another_strahlkorper) noexcept
    : l_max_(another_strahlkorper.l_max_),
      m_max_(another_strahlkorper.m_max_),
      ylm_(std::move(another_strahlkorper.ylm_)),
      // clang-tidy: do not std::move trivially constructable types
      center_(std::move(another_strahlkorper.center_)),  // NOLINT
      strahlkorper_coefs_(std::move(coefs)) {
  ASSERT(strahlkorper_coefs_.size() == ylm_.spectral_size(),
         "Bad size " << strahlkorper_coefs_.size() << ", expected "
                     << ylm_.spectral_size());
}

template <typename Frame>
void Strahlkorper<Frame>::pup(PUP::er& p) noexcept {
  p | l_max_;
  p | m_max_;
  p | center_;
  p | strahlkorper_coefs_;

  if (p.isUnpacking()) {
    ylm_ = YlmSpherepack(l_max_, m_max_);
  }
}

template <typename Frame>
std::array<double, 3> Strahlkorper<Frame>::physical_center() const noexcept {
  // Uses Eqs. (38)-(40) in Hemberger et al, arXiv:1211.6079.  This is
  // an approximation of Eq. (37) in the same paper, which gives the
  // exact result.
  std::array<double, 3> result = center_;
  SpherepackIterator it(l_max_, m_max_);
  result[0] += strahlkorper_coefs_[it.set(1, 1)()] * sqrt(0.75);
  result[1] -= strahlkorper_coefs_[it.set(1, -1)()] * sqrt(0.75);
  result[2] += strahlkorper_coefs_[it.set(1, 0)()] * sqrt(0.375);
  return result;
}

template <typename Frame>
double Strahlkorper<Frame>::average_radius() const noexcept {
  return ylm_.average(coefficients());
}

template <typename Frame>
double Strahlkorper<Frame>::radius(const double theta, const double phi) const
    noexcept {
  return ylm_.interpolate_from_coefs(strahlkorper_coefs_, {{{theta, phi}}})[0];
}

template <typename Frame>
bool Strahlkorper<Frame>::point_is_contained(
    const std::array<double, 3>& x) const noexcept {
  // The point `x` is assumed to be in Cartesian coords in the
  // Strahlkorper frame.

  // Make the point relative to the center of the Strahlkorper.
  auto xmc = x - center_;

  // Is the point inside the surface?
  const double theta = atan2(std::hypot(xmc[0], xmc[1]), xmc[2]);
  const double phi = atan2(xmc[1], xmc[0]);
  // Note that atan2 returns phi in (-pi,pi], whereas our convention
  // for spherical coordinates assumes phi is in [0,2pi).  In some
  // contexts (e.g. matching to points in a spherical-coordinate
  // element) this would cause problems, so we would need to add 2pi
  // to phi if phi were negative.  But here we don't need to do this,
  // because inside the 'radius' function, the only thing that phi is
  // used for is computing cos(m*phi) and sin(m*phi) for integer m.
  return magnitude(xmc) < radius(theta, phi);
}

template class Strahlkorper<Frame::Inertial>;
