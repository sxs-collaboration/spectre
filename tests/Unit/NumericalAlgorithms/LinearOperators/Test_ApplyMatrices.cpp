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
#include "Domain/Mesh.hpp"
#include "NumericalAlgorithms/LinearOperators/ApplyMatrices.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/ConstantExpressions.hpp"
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
    const Mesh<Dim>& mesh, const Index<Dim>& powers) noexcept {
  const auto coords = logical_coordinates(mesh);
  Variables<tmpl::list<ScalarTag, VectorTag>> result(
      mesh.number_of_grid_points(), 1.);
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
      const Mesh<Dim>& source_mesh, const Mesh<Dim>& dest_mesh,
      const Index<Dim>& powers,
      std::array<Matrix, Dim> matrices = std::array<Matrix, Dim>{}) noexcept {
    if (source_mesh.extents(FilledDim) == dest_mesh.extents(FilledDim)) {
      // Check implicit identity
      CheckApply<Dim, FilledDim + 1>::apply(source_mesh, dest_mesh, powers,
                                            matrices);
    }
    matrices[FilledDim] = Spectral::interpolation_matrix(
        source_mesh.slice_through(FilledDim),
        Spectral::collocation_points(dest_mesh.slice_through(FilledDim)));
    CheckApply<Dim, FilledDim + 1>::apply(source_mesh, dest_mesh, powers,
                                          matrices);
  }
};

template <size_t Dim>
struct CheckApply<Dim, Dim> {
  static void apply(const Mesh<Dim>& source_mesh, const Mesh<Dim>& dest_mesh,
                    const Index<Dim>& powers,
                    const std::array<Matrix, Dim>& matrices = {}) noexcept {
    const auto source_data = polynomial_data(source_mesh, powers);
    const auto result =
        apply_matrices(matrices, source_data, source_mesh.extents());
    const auto expected = polynomial_data(dest_mesh, powers);
    // Using this over CHECK_ITERABLE_APPROX speeds the test up by a
    // factor of 6 or so.
    for (const auto& p : result - expected) {
      CHECK(approx(p) == 0.);
    }
    const auto ref_matrices =
        make_array<std::reference_wrapper<const Matrix>, Dim>(matrices);
    CHECK(apply_matrices(ref_matrices, source_data, source_mesh.extents()) ==
          result);
    const auto datavector_result = apply_matrices(
        matrices, get(get<ScalarTag>(source_data)), source_mesh.extents());
    for (const auto& p : datavector_result - get(get<ScalarTag>(expected))) {
      CHECK(approx(p) == 0.);
    }
    CHECK(apply_matrices(ref_matrices, get(get<ScalarTag>(source_data)),
                         source_mesh.extents()) == datavector_result);
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
        Mesh<Dim> source_mesh{(*source_extents).indices(), basis, quadrature};
        Mesh<Dim> dest_mesh{(*dest_extents).indices(), basis, quadrature};
        CheckApply<Dim>::apply(std::move(source_mesh), std::move(dest_mesh),
                               *powers);
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

  // Can't use test_interpolation for 0 because Tensor errors on
  // Dim=0.
  const Index<0> extents{};
  Variables<tmpl::list<ScalarTag, VectorTag>> data(extents.product());
  get(get<ScalarTag>(data)) = DataVector{2.};
  get<0>(get<VectorTag>(data)) = DataVector{3.0};
  get<1>(get<VectorTag>(data)) = DataVector{4.0};
  const std::array<Matrix, 0> matrices{};
  // Can't construct the array directly because of
  // https://bugs.llvm.org/show_bug.cgi?id=35491 .
  // make_array contains a workaround.
  const std::array<std::reference_wrapper<const Matrix>, 0> ref_matrices =
      make_array<0, std::reference_wrapper<const Matrix>>(
          cpp17::as_const(Matrix{}));

  CHECK(apply_matrices(matrices, data, extents) == data);
  CHECK(apply_matrices(ref_matrices, data, extents) == data);
  const DataVector& data_vector = get(get<ScalarTag>(data));
  CHECK(apply_matrices(matrices, data_vector, extents) == data_vector);
  CHECK(apply_matrices(ref_matrices, data_vector, extents) == data_vector);
}
