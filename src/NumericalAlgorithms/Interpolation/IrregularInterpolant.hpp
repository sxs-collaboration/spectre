// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Utilities/Blas.hpp"
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
/// Interpolates a `Variables` onto an arbitrary set of points.
template <size_t Dim>
class Irregular {
 public:
  Irregular(
      const Mesh<Dim>& source_mesh,
      const tnsr::I<DataVector, Dim, Frame::Logical>& target_points) noexcept;
  Irregular();

  // clang-tidy: no runtime references
  /// Serialization for Charm++
  void pup(PUP::er& p) noexcept;  // NOLINT

  //@{
  /// Performs the interpolation on a `Variables` with grid points corresponding
  /// to the `Mesh<Dim>` specified in the constructor.
  /// The result is a `Variables` whose internal `DataVector` goes over the
  /// list of target_points that were specified in the constructor.
  /// \note for the void function, `result` will be resized to the proper size.
  template <typename TagsList>
  void interpolate(gsl::not_null<Variables<TagsList>*> result,
                   const Variables<TagsList>& vars) const noexcept;
  template <typename TagsList>
  Variables<TagsList> interpolate(const Variables<TagsList>& vars) const
      noexcept;
  //@}

 private:
  friend bool operator==(const Irregular& lhs, const Irregular& rhs) noexcept {
    return lhs.interpolation_matrix_ == rhs.interpolation_matrix_;
  }
  Matrix interpolation_matrix_;
};

template <size_t Dim>
template <typename TagsList>
void Irregular<Dim>::interpolate(
    const gsl::not_null<Variables<TagsList>*> result,
    const Variables<TagsList>& vars) const noexcept {
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
  dgemm_('n', 'n', m, n, k, 1.0, interpolation_matrix_.data(), m, vars.data(),
         k, 0.0, result->data(), m);
}

template <size_t Dim>
template <typename TagsList>
Variables<TagsList> Irregular<Dim>::interpolate(
    const Variables<TagsList>& vars) const noexcept {
  Variables<TagsList> result;
  interpolate(make_not_null(&result), vars);
  return result;
}

template <size_t Dim>
bool operator!=(const Irregular<Dim>& lhs,
                const Irregular<Dim>& rhs) noexcept;

}  // namespace intrp

