// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <functional>
#include <random>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/IndexIterator.hpp"
#include "DataStructures/Matrix.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/GetFundamentalType.hpp"

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

template <typename T>
using ScalarType = Scalar<T>;

template <typename T>
using VectorType = tnsr::I<T, 2>;

struct ScalarTag {
  using type = ScalarType<DataVector>;
};

struct TensorTag {
  using type = VectorType<DataVector>;
};

struct ComplexScalarTag {
  using type = ScalarType<ComplexDataVector>;
};

struct ComplexTensorTag {
  using type = VectorType<ComplexDataVector>;
};

template <typename LocalScalarTag, typename LocalTensorTag, size_t Dim>
Variables<tmpl::list<LocalScalarTag, LocalTensorTag>> polynomial_data(
    const Mesh<Dim>& mesh, const Index<Dim>& powers,
    const typename LocalScalarTag::type::type::ElementType
        fill_value) noexcept {
  const auto coords = logical_coordinates(mesh);
  Variables<tmpl::list<LocalScalarTag, LocalTensorTag>> result(
      mesh.number_of_grid_points(), fill_value);
  for (size_t i = 0; i < Dim; ++i) {
    get(get<LocalScalarTag>(result)) *= pow(coords.get(i), powers[i]);
    get<0>(get<LocalTensorTag>(result)) *= 2.0 * pow(coords.get(i), powers[i]);
    get<1>(get<LocalTensorTag>(result)) *= 3.0 * pow(coords.get(i), powers[i]);
  }
  return result;
}

template <typename LocalScalarTag, typename LocalTensorTag, size_t Dim,
          size_t FilledDim = 0>
struct CheckApply {
  static void apply(
      const Mesh<Dim>& source_mesh, const Mesh<Dim>& dest_mesh,
      const Index<Dim>& powers,
      std::array<Matrix, Dim> matrices = std::array<Matrix, Dim>{}) noexcept {
    if (source_mesh.extents(FilledDim) == dest_mesh.extents(FilledDim)) {
      // Check implicit identity
      CheckApply<LocalScalarTag, LocalTensorTag, Dim, FilledDim + 1>::apply(
          source_mesh, dest_mesh, powers, matrices);
    }
    matrices[FilledDim] = Spectral::interpolation_matrix(
        source_mesh.slice_through(FilledDim),
        Spectral::collocation_points(dest_mesh.slice_through(FilledDim)));
    CheckApply<LocalScalarTag, LocalTensorTag, Dim, FilledDim + 1>::apply(
        source_mesh, dest_mesh, powers, matrices);
  }
};

template <typename LocalScalarTag, typename LocalTensorTag, size_t Dim>
struct CheckApply<LocalScalarTag, LocalTensorTag, Dim, Dim> {
  static void apply(const Mesh<Dim>& source_mesh, const Mesh<Dim>& dest_mesh,
                    const Index<Dim>& powers,
                    const std::array<Matrix, Dim>& matrices = {}) noexcept {
    MAKE_GENERATOR(gen);
    UniformCustomDistribution<
        tt::get_fundamental_type_t<typename LocalScalarTag::type::type>>
        dist{0.1, 5.0};
    const auto fill_value = make_with_random_values<
        typename LocalScalarTag::type::type::ElementType>(make_not_null(&gen),
                                                          make_not_null(&dist));
    const auto source_data = polynomial_data<LocalScalarTag, LocalTensorTag>(
        source_mesh, powers, fill_value);
    const auto result =
        apply_matrices(matrices, source_data, source_mesh.extents());
    const auto expected = polynomial_data<LocalScalarTag, LocalTensorTag>(
        dest_mesh, powers, fill_value);
    // Using this over CHECK_ITERABLE_APPROX speeds the test up by a
    // factor of 6 or so.
    for (const auto& p : result - expected) {
      CHECK_COMPLEX_APPROX(p, 0.0);
    }
    const auto ref_matrices =
        make_array<std::reference_wrapper<const Matrix>, Dim>(matrices);
    CHECK(apply_matrices(ref_matrices, source_data, source_mesh.extents()) ==
          result);
    const auto vector_result = apply_matrices(
        matrices, get(get<LocalScalarTag>(source_data)), source_mesh.extents());
    for (const auto& p : vector_result - get(get<LocalScalarTag>(expected))) {
      CHECK_COMPLEX_APPROX(p, 0.0);
    }
    CHECK(apply_matrices(ref_matrices, get(get<LocalScalarTag>(source_data)),
                         source_mesh.extents()) == vector_result);
  }
};

template <typename LocalScalarTag, typename LocalTensorTag, size_t Dim>
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
        CheckApply<LocalScalarTag, LocalTensorTag, Dim>::apply(
            std::move(source_mesh), std::move(dest_mesh), *powers);
      }
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.ApplyMatrices",
                  "[DataStructures][Unit]") {
  {
    INFO("DataVector test");
    test_interpolation<ScalarTag, TensorTag, 1>();
    test_interpolation<ScalarTag, TensorTag, 2>();
    test_interpolation<ScalarTag, TensorTag, 3>();
  }
  {
    INFO("ComplexDataVector test");
    test_interpolation<ComplexScalarTag, ComplexTensorTag, 1>();
    test_interpolation<ComplexScalarTag, ComplexTensorTag, 2>();
    test_interpolation<ComplexScalarTag, ComplexTensorTag, 3>();
  }
  // Can't use test_interpolation for 0 because Tensor errors on
  // Dim=0.
  const Index<0> extents{};
  Variables<tmpl::list<ScalarTag, TensorTag>> data(extents.product());
  get(get<ScalarTag>(data)) = DataVector{2.0};
  get<0>(get<TensorTag>(data)) = DataVector{3.0};
  get<1>(get<TensorTag>(data)) = DataVector{4.0};

  Variables<tmpl::list<ComplexScalarTag, ComplexTensorTag>> complex_data(
      extents.product());
  get(get<ComplexScalarTag>(complex_data)) =
      ComplexDataVector{std::complex<double>{2.0, 3.0}};
  get<0>(get<ComplexTensorTag>(complex_data)) =
      ComplexDataVector{std::complex<double>{3.0, 4.0}};
  get<1>(get<ComplexTensorTag>(complex_data)) =
      ComplexDataVector{std::complex<double>{4.0, 5.0}};

  const std::array<Matrix, 0> matrices{};
  // Can't construct the array directly because of
  // https://bugs.llvm.org/show_bug.cgi?id=35491 .
  // make_array contains a workaround.
  const std::array<std::reference_wrapper<const Matrix>, 0> ref_matrices =
      make_array<0, std::reference_wrapper<const Matrix>>(
          std::add_const_t<Matrix>{});

  CHECK(apply_matrices(matrices, data, extents) == data);
  CHECK(apply_matrices(ref_matrices, data, extents) == data);

  CHECK(apply_matrices(matrices, complex_data, extents) == complex_data);
  CHECK(apply_matrices(ref_matrices, complex_data, extents) == complex_data);

  const DataVector& data_vector = get(get<ScalarTag>(data));
  CHECK(apply_matrices(matrices, data_vector, extents) == data_vector);
  CHECK(apply_matrices(ref_matrices, data_vector, extents) == data_vector);

  const ComplexDataVector& complex_data_vector =
      get(get<ComplexScalarTag>(complex_data));
  CHECK(apply_matrices(matrices, complex_data_vector, extents) ==
        complex_data_vector);
  CHECK(apply_matrices(ref_matrices, complex_data_vector, extents) ==
        complex_data_vector);
}
