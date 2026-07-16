// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstdint>
#include <iosfwd>
#include <string>

#include "Utilities/MakeArray.hpp"

/// \cond
namespace Options {
class Option;
template <typename T>
struct create_from_yaml;
}  // namespace Options
/// \endcond

namespace Spectral {
/// \brief The amount we left-shift the basis integers by.
///
/// We do this to be able to represent the combined Basis and Quadrature as one
/// 8-bit integer.
constexpr uint8_t basis_shift = 4;
/*!
 * \brief Either the basis functions used by a spectral or discontinuous
 * Galerkin (DG) method, or the value `FiniteDifference` when a finite
 * difference method is used
 *
 * \details The particular choices of Basis and Quadrature determine where the
 * collocation points of a Mesh are located in an Element. For a spectral or DG
 * method, the Basis also represents the choice of basis functions used to
 * represent a function on an Element, which then provides a convenient choice
 * for the operators used for differentiation, interpolation, etc.  For a finite
 * difference method, one needs to choose the order of the scheme (and hence the
 * weights, differentiation matrix, integration weights, and interpolant)
 * locally in space and time to handle discontinuous solutions.
 *
 * \note Choose `Legendre` for a general-purpose spectral or DG mesh, unless
 * you have a particular reason for choosing `Chebyshev`.
 *
 * \note Choose `Fourier` for a dimension that is spatially periodic and in
 * which the expected solution is smooth.  A Mesh with this Basis cannot be
 * split by h-refinement in this dimension.
 *
 * \note Choose two consecutive dimensions to have `SphericalHarmonic` to choose
 * a spherical harmonic basis.  By convention, the first dimension represents
 * the polar/zenith angle (or colatitude), while the second dimension represents
 * the azimuthal angle (or longitude).  A Mesh with this Basis cannot be split
 * by h-refinement in these dimensions.
 *
 * \note Choose two consecutive dimensions to have `ZernikeB2` to choose
 * a basis used on a disk or cross-section of a cylinder.  By convention, the
 * first dimension represents the radial direction, while the second dimension
 * represents the azimuthal angle. A Mesh with this Basis cannot be split by
 * h-refinement in these dimensions.
 *
 * \note Choose three consecutive dimensions to have `ZernikeB3` to choose
 * a basis used on a sphere.  By convention, the first dimension represents the
 * radial direction, the second dimension represents the polar/zenith angle (or
 * colatitude), while the third dimension represents the azimuthal angle (or
 * longitude). A Mesh with this Basis cannot be split by h-refinement in these
 * dimensions.
 *
 * \note Choose `Cartoon` for a dimension (or consecutive dimensions) that
 * represent axial (or spherical) symmetry.
 *
 * \remark We store these effectively as a 4-bit integer using the highest 4
 * bits of a uint8_t, which is why we do the left shift.  We cannot have more
 * than 16 bases to fit into the 4 bits, including the `Uninitialized` value.
 * The number of bits to shift is encoded in the variable
 * `Spectral::detail::basis_shift`.
 */
enum class Basis : uint8_t {
  Uninitialized = 0 << basis_shift,
  Chebyshev = 1 << basis_shift,
  Legendre = 2 << basis_shift,
  FiniteDifference = 3 << basis_shift,
  SphericalHarmonic = 4 << basis_shift,
  Fourier = 5 << basis_shift,
  ZernikeB1 = 6 << basis_shift,
  ZernikeB2 = 7 << basis_shift,
  ZernikeB3 = 8 << basis_shift,
  Cartoon = 9 << basis_shift
};

/// All possible values of Basis
std::array<Basis, 10> all_bases();

/// Convert a string to a Basis enum.
Basis to_basis(const std::string& basis);

/// Output operator for a Basis.
std::ostream& operator<<(std::ostream& os, const Basis& basis);

/// Shortcuts for common Basis products where the default for I1Basis is
/// Basis::Legendre
namespace bases {
template <size_t VolumeDim, Basis I1Basis = Basis::Legendre>
static constexpr auto hypercube = make_array<VolumeDim>(Basis::Legendre);

template <size_t VolumeDim>
static constexpr auto hypertorus = make_array<VolumeDim>(Basis::Fourier);

template <Basis I1Basis = Basis::Legendre>
static constexpr auto annulus = std::array{I1Basis, Basis::Fourier};

static constexpr auto spherical_surface =
    make_array<2>(Basis::SphericalHarmonic);

static constexpr auto disk = make_array<2>(Basis::ZernikeB2);

template <Basis I1Basis = Basis::Legendre>
static constexpr auto spherical_shell =
    std::array{I1Basis, Basis::SphericalHarmonic, Basis::SphericalHarmonic};

template <Basis I1Basis = Basis::Legendre>
static constexpr auto cylindrical_shell =
    std::array{I1Basis, Basis::Fourier, I1Basis};

template <Basis I1Basis = Basis::Legendre>
static constexpr auto full_cylinder =
    std::array{Basis::ZernikeB2, Basis::ZernikeB2, I1Basis};

static constexpr auto full_sphere = make_array<3>(Basis::ZernikeB3);

template <Basis I1Basis = Basis::Legendre>
static constexpr auto cartoon_sphere =
    std::array{I1Basis, Basis::Cartoon, Basis::Cartoon};

static constexpr auto cartoon_sphere_inner =
    std::array{Basis::ZernikeB1, Basis::Cartoon, Basis::Cartoon};

template <Basis I1Basis = Basis::Legendre>
static constexpr auto cartoon_cylinder =
    std::array{I1Basis, I1Basis, Basis::Cartoon};

template <Basis I1Basis = Basis::Legendre>
static constexpr auto cartoon_cylinder_inner =
    std::array{Basis::ZernikeB1, I1Basis, Basis::Cartoon};
}  // namespace bases
}  // namespace Spectral

template <>
struct Options::create_from_yaml<Spectral::Basis> {
  template <typename Metavariables>
  static Spectral::Basis create(const Options::Option& options) {
    return create<void>(options);
  }
};

template <>
Spectral::Basis Options::create_from_yaml<Spectral::Basis>::create<void>(
    const Options::Option& options);
