// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
class DataVector;
template <size_t Dim>
class Mesh;
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace intrp {
/*!
 * \brief Interpolates by doing partial summation in each dimension using
 * one-dimensional interpolation
 *
 * \details The one-dimensional matrices used to do the interpolation depend
 * upon the Spectral::Basis used in each dimension:
 * - For a Chebyshev or Legendre basis, the matrices are given by
 * Spectral::fornberg_interpolation_matrix at the quadrature points of the
 * source_mesh.  (These are equivalent to those returned by
 * Spectral::interpolation_matrix.)
 * - For a Fourier basis, the matrix is given by
 * Spectral::fourier_interpolation_matrix at the quadrature points of the
 * source_mesh
 *
 * For multidimensional bases such as the SphericalHarmonic or ZernikeB2, the
 * matrices used for interpolating cannot be applied per dimension but must
 * be handled specially.
 *
 */
template <size_t Dim>
class Cardinal {
 public:
  Cardinal(
      const Mesh<Dim>& source_mesh,
      const tnsr::I<DataVector, Dim, Frame::ElementLogical>& target_points);
  Cardinal(const Mesh<Dim>& source_mesh,
           const tnsr::I<double, Dim, Frame::ElementLogical>& target_point);

  Cardinal();

  /// Interpolates the function `f` provided on the `source_mesh` to the
  /// `target_points` with which the interpolator was constructed.
  DataVector interpolate(const DataVector& f) const;

  /// The one-dimensional interpolation matrices used to do the interpolation
  const std::array<Matrix, Dim>& interpolation_matrices() const;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

 private:
  /// Precomputes `zernike_weights_`, which is all the work independent of
  /// `f_source`, to avoid redundant computations. This is only needed when
  /// the source mesh has a B2 basis
  void set_zernike_b2_weights();

  /// General routine called by `interpolate()` for a mesh using B2 bases
  DataVector interpolate_zernike_b2(const DataVector& f_source) const;

  template <size_t LocalDim>
  // NOLINTNEXTILNE(readability-redundant-declaration)
  friend bool operator==(const Cardinal<LocalDim>& lhs,
                         const Cardinal<LocalDim>& rhs);

  size_t n_target_points_ = 0;
  Mesh<Dim> source_mesh_{};
  std::array<Matrix, Dim> interpolation_matrices_{};
  bool using_spherical_harmonics_{false};
  bool using_zernike_b2_{false};
  Matrix zernike_weights_{};
};

template <size_t Dim>
bool operator==(const Cardinal<Dim>& lhs, const Cardinal<Dim>& rhs);

template <size_t Dim>
bool operator!=(const Cardinal<Dim>& lhs, const Cardinal<Dim>& rhs);
}  // namespace intrp
