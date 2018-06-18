// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/IndexIterator.hpp"
#include "DataStructures/Matrix.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/LinearOperators/ApplyMatrices.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"

namespace {
constexpr Spectral::Basis basis = Spectral::Basis::Legendre;
// Using Gauss points for this test would be nice, since that would
// let us test the extent=1 cases, but that needs to wait for
// everything to be updated to use Mesh.
constexpr Spectral::Quadrature quadrature = Spectral::Quadrature::GaussLobatto;
template <size_t Dim>
constexpr size_t max_points = 4;
template <>
constexpr size_t max_points<3> = 3;

using ScalarType = Scalar<DataVector>;
using VectorType = tnsr::I<DataVector, 2>;

struct ScalarTag {
  using type = ScalarType;
};

struct VectorTag {
  using type = VectorType;
};

template <size_t Dim>
Variables<tmpl::list<ScalarTag, VectorTag>> polynomial_data(
    const Index<Dim>& extents, const Index<Dim>& powers) noexcept {
  const auto coords = logical_coordinates(extents);
  Variables<tmpl::list<ScalarTag, VectorTag>> result(extents.product(), 1.);
  for (size_t i = 0; i < Dim; ++i) {
    get(get<ScalarTag>(result)) *= pow(coords.get(i), powers[i]);
    get<0>(get<VectorTag>(result)) *= 2.0 * pow(coords.get(i), powers[i]);
    get<1>(get<VectorTag>(result)) *= 3.0 * pow(coords.get(i), powers[i]);
  }
  return result;
}

template <size_t Dim, size_t FilledDim = 0>
struct CheckApply {
  static void apply(
      const Index<Dim>& source_extents, const Index<Dim>& dest_extents,
      const Index<Dim>& powers,
      std::array<Matrix, Dim> matrices = std::array<Matrix, Dim>{}) noexcept {
    if (source_extents[FilledDim] == dest_extents[FilledDim]) {
      // Check implicit identity
      CheckApply<Dim, FilledDim + 1>::apply(source_extents, dest_extents,
                                            powers, matrices);
    }
    matrices[FilledDim] = Spectral::interpolation_matrix<basis, quadrature>(
        source_extents[FilledDim],
        Spectral::collocation_points<basis, quadrature>(
            dest_extents[FilledDim]));
    CheckApply<Dim, FilledDim + 1>::apply(source_extents, dest_extents, powers,
                                          matrices);
  }
};

template <size_t Dim>
struct CheckApply<Dim, Dim> {
  static void apply(const Index<Dim>& source_extents,
                    const Index<Dim>& dest_extents, const Index<Dim>& powers,
                    const std::array<Matrix, Dim>& matrices = {}) noexcept {
    const auto source_data = polynomial_data(source_extents, powers);
    const auto result = apply_matrices(matrices, source_data, source_extents);
    const auto expected = polynomial_data(dest_extents, powers);
    // Using this over CHECK_ITERABLE_APPROX speeds the test up by a
    // factor of 6 or so.
    for (const auto& p : result - expected) {
      CHECK(approx(p) == 0.);
    }
    const auto ref_matrices =
        make_array<std::reference_wrapper<const Matrix>, Dim>(matrices);
    CHECK(apply_matrices(ref_matrices, source_data, source_extents) == result);
    const auto datavector_result = apply_matrices(
        matrices, get(get<ScalarTag>(source_data)), source_extents);
    for (const auto& p : datavector_result - get(get<ScalarTag>(expected))) {
      CHECK(approx(p) == 0.);
    }
    CHECK(apply_matrices(ref_matrices, get(get<ScalarTag>(source_data)),
                         source_extents) == datavector_result);
  }
};

template <size_t Dim>
void test_interpolation() noexcept {
  const auto too_few_points = [](const size_t extent) noexcept {
    return extent < Spectral::minimum_number_of_points<basis, quadrature>;
  };

  for (IndexIterator<Dim> source_extents(Index<Dim>(max_points<Dim> + 1));
       source_extents; ++source_extents) {
    if (std::any_of(source_extents->begin(), source_extents->end(),
                    too_few_points)) {
      continue;
    }
    CAPTURE(*source_extents);
    for (IndexIterator<Dim> dest_extents(Index<Dim>(max_points<Dim> + 1));
         dest_extents; ++dest_extents) {
      if (std::any_of(dest_extents->begin(), dest_extents->end(),
                      too_few_points)) {
        continue;
      }
      CAPTURE(*dest_extents);
      Index<Dim> max_powers;
      for (size_t i = 0; i < Dim; ++i) {
        max_powers[i] = std::min((*source_extents)[i], (*dest_extents)[i]);
      }
      for (IndexIterator<Dim> powers(max_powers); powers; ++powers) {
        CAPTURE(*powers);
        CheckApply<Dim>::apply(*source_extents, *dest_extents, *powers);
      }
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.LinearOperators.ApplyMatrices",
                  "[NumericalAlgorithms][LinearOperators][Unit]") {
  test_interpolation<1>();
  test_interpolation<2>();
  test_interpolation<3>();
}
