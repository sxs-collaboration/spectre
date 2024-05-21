
// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <iosfwd>
#include <string>

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
 * \details The particular choices of Basis and Quadrature determine
 * where the collocation points of a Mesh are located in an Element.
 * For a spectral or DG method, the Basis also represents the choice
 * of basis functions used to represent a function on an Element,
 * which then provides a convenient choice for the operators used for
 * differentiation, interpolation, etc.  For a finite difference
 * method, one needs to choose the order of the scheme (and hence the
 * weights, differentiation matrix, integration weights, and
 * interpolant) locally in space and time to handle discontinuous
 * solutions.
 *
 * \note Choose `Legendre` for a general-purpose spectral or DG mesh, unless
 * you have a particular reason for choosing `Chebyshev`.
 *
 * \note Choose two consecutive dimensions to have `SphericalHarmonic` to choose
 * a spherical harmonic basis.  By convention, the first dimension represents
 * the polar/zentith angle (or colatitude), while the second dimension
 * represents the azimuthal angle (or longitude)
 *
 * \note We store these effectively as a 4-bit integer using the highest 4
 * bits of a uint8_t, which is why we do the left shift.  We cannot
 * have more than 16 bases to fit into the 4 bits, including the
 * `Uninitialized` value. The number of bits to shift is encoded in the variable
 * `Spectral::detail::basis_shift`.
 */
enum class Basis : uint8_t {
  Uninitialized = 0 << basis_shift,
  Chebyshev = 1 << basis_shift,
  Legendre = 2 << basis_shift,
  FiniteDifference = 3 << basis_shift,
  SphericalHarmonic = 4 << basis_shift
};

/// All possible values of Basis
std::array<Basis, 5> all_bases();

/// Convert a string to a Basis enum.
Basis to_basis(const std::string& basis);

/// Output operator for a Basis.
std::ostream& operator<<(std::ostream& os, const Basis& basis);
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
