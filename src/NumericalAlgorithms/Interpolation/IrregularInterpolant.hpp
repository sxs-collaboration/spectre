// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
template <size_t Dim>
class Mesh;
namespace PUP {
class er;
}  // namespace PUP
// IWYU pragma: no_forward_declare Variables
/// \endcond

namespace intrp {

/// \ingroup NumericalAlgorithmsGroup
/// \brief Interpolates a `Variables` onto an arbitrary set of points.
///
/// \details If the `source_mesh` uses
/// SpatialDiscretization::Basis::FiniteDifference, linear interpolation is done
/// in each dimension; otherwise it uses the barycentric interpolation provided
/// by Spectral::interpolation_matrix in each dimension.
template <size_t Dim>
class Irregular {
 public:
  Irregular(
      const Mesh<Dim>& source_mesh,
      const tnsr::I<DataVector, Dim, Frame::ElementLogical>& target_points);
  Irregular();

  /// Serialization for Charm++
  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

  /// @{
  /// Performs the interpolation on a `Variables` with grid points corresponding
  /// to the `Mesh<Dim>` specified in the constructor.
  /// The result is a `Variables` whose internal `DataVector` goes over the
  /// list of target_points that were specified in the constructor.
  /// \note for the void function, `result` will be resized to the proper size.
  template <typename TagsList>
  void interpolate(gsl::not_null<Variables<TagsList>*> result,
                   const Variables<TagsList>& vars) const;
  template <typename TagsList>
  Variables<TagsList> interpolate(const Variables<TagsList>& vars) const;
  /// @}

  /// @{
  /// \brief Interpolate a DataVector onto the target points.
  ///
  /// \note When interpolating multiple tensors, the Variables interface is more
  /// efficient. However, this DataVector interface is useful for applications
  /// where only some components of a Tensor or Variables need to be
  /// interpolated.
  void interpolate(gsl::not_null<DataVector*> result,
                   const DataVector& input) const;
  DataVector interpolate(const DataVector& input) const;
  /// @}

 private:
  friend bool operator==(const Irregular& lhs, const Irregular& rhs) {
    return lhs.interpolation_matrix_ == rhs.interpolation_matrix_;
  }
  Matrix interpolation_matrix_;
};

template <size_t Dim>
template <typename TagsList>
void Irregular<Dim>::interpolate(
    const gsl::not_null<Variables<TagsList>*> result,
    const Variables<TagsList>& vars) const {
  // For matrix multiplication of Interp . Source = Result:
  //   matrix Interp is m rows by k columns
  //   matrix Source is k rows by n columns
  //   matrix Result is m rows by n columns
  const size_t m = interpolation_matrix_.rows();
  const size_t k = interpolation_matrix_.columns();
  const size_t n = vars.number_of_independent_components;
  ASSERT(k == vars.number_of_grid_points(),
         "Number of grid points in source 'vars', "
             << vars.number_of_grid_points()
             << ",\n disagrees with the size of the source_mesh, " << k
             << ", that was passed into the constructor");
  if (result->number_of_grid_points() != m) {
    *result = Variables<TagsList>(m, 0.);
  }
  dgemm_('n', 'n', m, n, k, 1.0, interpolation_matrix_.data(),
         interpolation_matrix_.spacing(), vars.data(), k, 0.0, result->data(),
         m);
}

template <size_t Dim>
template <typename TagsList>
Variables<TagsList> Irregular<Dim>::interpolate(
    const Variables<TagsList>& vars) const {
  Variables<TagsList> result;
  interpolate(make_not_null(&result), vars);
  return result;
}

template <size_t Dim>
bool operator!=(const Irregular<Dim>& lhs, const Irregular<Dim>& rhs);

}  // namespace intrp
