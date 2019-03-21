// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <boost/functional/hash/extensions.hpp>
#include <cstddef>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/ModalVector.hpp"
#include "DataStructures/Tags.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Direction.hpp"
#include "Domain/Element.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/OrientationMap.hpp"
#include "Evolution/DiscontinuousGalerkin/SlopeLimiters/Krivodonova.hpp"
#include "NumericalAlgorithms/LinearOperators/CoefficientTransforms.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace SlopeLimiters {

// The overall way of testing the Krivodonova limiter is to set the modal
// coefficients directly, then compare to the expected result from the algorithm
// for 3 collocation points in the documentation. In the 1D case it is easy
// enough to test with 4 collocation points, so this is done instead. In order
// to make sure the loops over tensor components work correctly we apply the
// limiter both to a Scalar and to a tnsr::I<Dim>.
namespace {
template <size_t VolumeDim, typename PackagedData>
using NeighborData = std::unordered_map<
    std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, PackagedData,
    boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>;

template <size_t Identifier>
struct ScalarTag : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <size_t Dim, size_t Identifier>
struct VectorTag : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim>;
};

namespace test_1d {
void test_package_data(const size_t order) noexcept {
  INFO("Testing package data");
  CAPTURE(order);
  const Mesh<1> mesh(order + 1, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);
  const DataVector& x = Spectral::collocation_points(mesh);
  using Limiter = Krivodonova<1, tmpl::list<ScalarTag<0>, VectorTag<1, 0>>>;
  Limiter krivodonova{};

  Scalar<DataVector> tensor0(mesh.number_of_grid_points());
  get(tensor0) =
      Spectral::compute_basis_function_value<Spectral::Basis::Legendre>(order,
                                                                        x);
  tnsr::I<DataVector, 1> tensor1(mesh.number_of_grid_points());
  get<0>(tensor1) =
      Spectral::compute_basis_function_value<Spectral::Basis::Legendre>(order,
                                                                        x) +
      2.0 * Spectral::compute_basis_function_value<Spectral::Basis::Legendre>(
                order - 1, x);
  Limiter::PackagedData packaged_data{};

  // test no reorienting
  {
    krivodonova.package_data(make_not_null(&packaged_data), tensor0, tensor1,
                             mesh, {});
    ModalVector expected0(mesh.number_of_grid_points(), 0.0);
    expected0[mesh.number_of_grid_points() - 1] = 1.0;
    CHECK(get(get<::Tags::Modal<ScalarTag<0>>>(
              packaged_data.modal_volume_data)) == expected0);
    ModalVector expected1(mesh.number_of_grid_points(), 0.0);
    expected1[mesh.number_of_grid_points() - 1] = 1.0;
    expected1[mesh.number_of_grid_points() - 2] = 2.0;
    CHECK(get<0>(get<::Tags::Modal<VectorTag<1, 0>>>(
              packaged_data.modal_volume_data)) == expected1);
  }
  // test reorienting
  {
    krivodonova.package_data(
        make_not_null(&packaged_data), tensor0, tensor1, mesh,
        {{{Direction<1>::upper_xi()}}, {{Direction<1>::lower_xi()}}});
    ModalVector expected0(mesh.number_of_grid_points(), 0.0);
    expected0[0] = 1.0;
    CHECK(get(get<::Tags::Modal<ScalarTag<0>>>(
              packaged_data.modal_volume_data)) == expected0);
    ModalVector expected1(mesh.number_of_grid_points(), 0.0);
    expected1[0] = 1.0;
    expected1[1] = 2.0;
    CHECK(get<0>(get<::Tags::Modal<VectorTag<1, 0>>>(
              packaged_data.modal_volume_data)) == expected1);
  }
}

void test_limiting_two_neighbors() noexcept {
  INFO("Testing applying limiter to coefficients");
  constexpr size_t dim = 1;
  const size_t order = 3;
  const Mesh<dim> mesh(order + 1, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto);
  const size_t num_pts = mesh.number_of_grid_points();

  using Limiter = Krivodonova<dim, tmpl::list<ScalarTag<0>, VectorTag<dim, 0>>>;
  // Use non-unity (because that's the default) but close alpha values to make
  // the math easier but still test thoroughly.
  Limiter krivodonova{
      make_array<Spectral::maximum_number_of_points<Spectral::Basis::Legendre>>(
          0.99)};

  NeighborData<dim, typename Limiter::PackagedData> neighbor_data{};

  const Element<dim> element(ElementId<dim>{0}, {});
  // We don't care about the ElementId for these tests, just the direction.
  Limiter::PackagedData& package_data_upper = neighbor_data[std::make_pair(
      Direction<dim>::upper_xi(), ElementId<dim>{0})];
  Limiter::PackagedData& package_data_lower = neighbor_data[std::make_pair(
      Direction<dim>::lower_xi(), ElementId<dim>{0})];

  package_data_upper.modal_volume_data.initialize(num_pts);
  package_data_upper.mesh = mesh;
  package_data_lower.modal_volume_data.initialize(num_pts);
  package_data_lower.mesh = mesh;

  Scalar<DataVector> nodal_scalar_data_to_limit(num_pts, 0.0);
  tnsr::I<DataVector, dim> nodal_vector_data_to_limit(num_pts, 0.0);
  DataVector expected(num_pts);
  const auto helper =
      [
        &element, &expected, &krivodonova, &mesh, &neighbor_data,
        &package_data_lower, &package_data_upper, &nodal_scalar_data_to_limit, &
        nodal_vector_data_to_limit
      ](const ModalVector& upper_coeffs, const ModalVector& initial_coeffs,
        const ModalVector& lower_coeffs,
        const ModalVector& expected_coeffs) noexcept {
    to_nodal_coefficients(&get(nodal_scalar_data_to_limit), initial_coeffs,
                          mesh);
    get(get<::Tags::Modal<ScalarTag<0>>>(
        package_data_upper.modal_volume_data)) = upper_coeffs;
    get(get<::Tags::Modal<ScalarTag<0>>>(
        package_data_lower.modal_volume_data)) = lower_coeffs;
    for (size_t i = 0; i < dim; ++i) {
      to_nodal_coefficients(&nodal_vector_data_to_limit.get(i), initial_coeffs,
                            mesh);
      get<::Tags::Modal<VectorTag<dim, 0>>>(
          package_data_upper.modal_volume_data)
          .get(i) = upper_coeffs;
      get<::Tags::Modal<VectorTag<dim, 0>>>(
          package_data_lower.modal_volume_data)
          .get(i) = lower_coeffs;
    }
    krivodonova(&nodal_scalar_data_to_limit, &nodal_vector_data_to_limit,
                element, mesh, neighbor_data);
    to_nodal_coefficients(&expected, expected_coeffs, mesh);
    CHECK_ITERABLE_APPROX(get(nodal_scalar_data_to_limit), expected);
    for (size_t i = 0; i < dim; ++i) {
      CHECK_ITERABLE_APPROX(nodal_vector_data_to_limit.get(i), expected);
    }
  };

  // Note: Throughout the tests 1.0e-18 is used anywhere that we need a positive
  // but zero coefficient, because it is below machine precision for O(1)
  // numbers.

  // Limit no coefficients because the highest coefficient doesn't need
  // limiting.
  helper({7.0, 3.0, 4.1, 0.0}, {3.0, -2.0, 2.0, 1.0}, {1.0, -5.0, 1.0e-18, 0.0},
         {3.0, -2.0, 2.0, 1.0});

  // Limit the highest coefficient only:
  // Limit top coeff from upper neighbor
  helper({7.0, 3.0, 2.5, 0.0}, {3.0, -2.0, 2.0, 1.0}, {1.0, -5.0, 1.0e-18, 0.0},
         {3.0, -2.0, 2.0, 0.5 * 0.99});
  // Limit top coeff from lower neighbor
  helper({7.0, 3.0, 4.1, 0.0}, {3.0, -2.0, 2.0, 1.0}, {1.0, -5.0, 1.3, 0.0},
         {3.0, -2.0, 2.0, 0.7 * 0.99});
  // Zero top coeff from upper neighbor
  helper({7.0, 3.0, 1.9, 0.0}, {3.0, -2.0, 2.0, 1.0}, {1.0, -5.0, 1.0e-18, 0.0},
         {3.0, -2.0, 2.0, 0.0});
  // Zero top coeff from lower neighbor
  helper({7.0, 3.0, 4.1, 0.0}, {3.0, -2.0, 2.0, 1.0}, {1.0, -5.0, 5.3, 0.0},
         {3.0, -2.0, 2.0, 0.0});

  // Limit the top 2 coeffs:
  // Limit top upper neighbor, limit second upper neighbor
  helper({7.0, 3.0, 2.5, 0.0}, {3.0, 2.0, 2.0, 1.0}, {-2.1, -5.0, 1.0e-18, 0.0},
         {3.0, 2.0, 1.0 * 0.99, 0.5 * 0.99});
  // Limit top upper neighbor, limit second lower neighbor
  helper({7.0, 5.0, 2.5, 0.0}, {3.0, 2.0, 2.0, 1.0}, {-2.1, 1.1, 1.0e-18, 0.0},
         {3.0, 2.0, 0.9 * 0.99, 0.5 * 0.99});
  // Limit top upper neighbor, zero second upper neighbor
  helper({7.0, 1.0, 2.5, 0.0}, {3.0, 2.0, 2.0, 1.0}, {-2.1, 1.1, 1.0e-18, 0.0},
         {3.0, 2.0, 0.0, 0.5 * 0.99});
  // Limit top upper neighbor, zero second lower neighbor
  helper({7.0, 5.0, 2.5, 0.0}, {3.0, 2.0, 2.0, 1.0}, {-2.1, 2.1, 1.0e-18, 0.0},
         {3.0, 2.0, 0.0, 0.5 * 0.99});
  // Zero top upper neighbor, limit second upper neighbor
  helper({7.0, 3.0, 1.9, 0.0}, {3.0, 2.0, 2.0, 1.0}, {-2.1, -5.0, 1.0e-18, 0.0},
         {3.0, 2.0, 1.0 * 0.99, 0.0});
  // Zero top upper neighbor, limit second lower neighbor
  helper({7.0, 5.0, 1.9, 0.0}, {3.0, 2.0, 2.0, 1.0}, {-2.1, 1.1, 1.0e-18, 0.0},
         {3.0, 2.0, 0.9 * 0.99, 0.0});
  // Zero top upper neighbor, zero second upper neighbor
  helper({7.0, 1.0, 1.9, 0.0}, {3.0, 2.0, 2.0, 1.0}, {-2.1, 1.1, 1.0e-18, 0.0},
         {3.0, 2.0, 0.0, 0.0});
  // Zero top upper neighbor, zero second lower neighbor
  helper({7.0, 5.0, 1.9, 0.0}, {3.0, 2.0, 2.0, 1.0}, {-2.1, 2.1, 1.0e-18, 0.0},
         {3.0, 2.0, 0.0, 0.0});

  // Limit top lower neighbor, limit second upper neighbor
  helper({7.0, 3.0, 4.1, 0.0}, {3.0, 2.0, 2.0, 1.0}, {-2.1, -5.0, 1.3, 0.0},
         {3.0, 2.0, 1.0 * 0.99, 0.7 * 0.99});
  // Limit top lower neighbor, limit second lower neighbor
  helper({7.0, 5.0, 4.1, 0.0}, {3.0, 2.0, 2.0, 1.0}, {-2.1, 1.1, 1.3, 0.0},
         {3.0, 2.0, 0.9 * 0.99, 0.7 * 0.99});
  // Limit top lower neighbor, zero second upper neighbor
  helper({7.0, 1.0, 4.1, 0.0}, {3.0, 2.0, 2.0, 1.0}, {-2.1, 1.1, 1.3, 0.0},
         {3.0, 2.0, 0.0, 0.7 * 0.99});
  // Limit top lower neighbor, zero second lower neighbor
  helper({7.0, 5.0, 4.1, 0.0}, {3.0, 2.0, 2.0, 1.0}, {-2.1, 2.1, 1.3, 0.0},
         {3.0, 2.0, 0.0, 0.7 * 0.99});
  // Zero top lower neighbor, limit second upper neighbor
  helper({7.0, 3.0, 4.1, 0.0}, {3.0, 2.0, 2.0, 1.0}, {-2.1, -5.0, 3.0, 0.0},
         {3.0, 2.0, 1.0 * 0.99, 0.0});
  // Zero top lower neighbor, limit second lower neighbor
  helper({7.0, 5.0, 4.1, 0.0}, {3.0, 2.0, 2.0, 1.0}, {-2.1, 1.1, 3.0, 0.0},
         {3.0, 2.0, 0.9 * 0.99, 0.0});
  // Zero top lower neighbor, zero second upper neighbor
  helper({7.0, 1.0, 4.1, 0.0}, {3.0, 2.0, 2.0, 1.0}, {-2.1, 1.1, 3.0, 0.0},
         {3.0, 2.0, 0.0, 0.0});
  // Zero top lower neighbor, zero second lower neighbor
  helper({7.0, 5.0, 4.1, 0.0}, {3.0, 2.0, 2.0, 1.0}, {-2.1, 2.1, 3.0, 0.0},
         {3.0, 2.0, 0.0, 0.0});
}

void test_limiting_different_values_different_tensors() noexcept {
  INFO("Testing different values for each tensor component");
  constexpr size_t dim = 1;
  const size_t order = 3;
  const Mesh<dim> mesh(order + 1, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto);
  const size_t num_pts = mesh.number_of_grid_points();
  using Limiter = Krivodonova<dim, tmpl::list<ScalarTag<0>, VectorTag<dim, 0>>>;
  // Use non-unity (because that's the default) but close alpha values to make
  // the math easier but still test thoroughly.
  Limiter krivodonova{
      make_array<Spectral::maximum_number_of_points<Spectral::Basis::Legendre>>(
          0.99)};

  NeighborData<dim, typename Limiter::PackagedData> neighbor_data{};

  const Element<dim> element(ElementId<dim>{0}, {});
  // We don't care about the ElementId for these tests, just the direction.
  Limiter::PackagedData& package_data_up_xi = neighbor_data[std::make_pair(
      Direction<dim>::upper_xi(), ElementId<dim>{0})];
  Limiter::PackagedData& package_data_lo_xi = neighbor_data[std::make_pair(
      Direction<dim>::lower_xi(), ElementId<dim>{0})];

  package_data_up_xi.modal_volume_data.initialize(num_pts);
  package_data_up_xi.mesh = mesh;
  package_data_lo_xi.modal_volume_data.initialize(num_pts);
  package_data_lo_xi.mesh = mesh;

  Scalar<DataVector> nodal_scalar_data_to_limit(num_pts, 0.0);
  tnsr::I<DataVector, dim> nodal_vector_data_to_limit(num_pts, 0.0);
  Scalar<DataVector> nodal_scalar_expected(num_pts, 0.0);
  tnsr::I<DataVector, dim> nodal_vector_expected(num_pts, 0.0);

  // Limit top coeff from upper neighbor
  to_nodal_coefficients(&get(nodal_scalar_data_to_limit), {3.0, -2.0, 2.0, 1.0},
                        mesh);
  get(get<::Tags::Modal<ScalarTag<0>>>(package_data_up_xi.modal_volume_data)) =
      ModalVector{7.0, 3.0, 2.5, 0.0};
  get(get<::Tags::Modal<ScalarTag<0>>>(package_data_lo_xi.modal_volume_data)) =
      ModalVector{1.0, -5.0, 1.0e-18, 0.0};
  to_nodal_coefficients(&get(nodal_scalar_expected),
                        {3.0, -2.0, 2.0, 0.5 * 0.99}, mesh);

  // Limit top upper neighbor, limit second lower neighbor
  to_nodal_coefficients(&nodal_vector_data_to_limit.get(0),
                        {3.0, 2.0, 2.0, 1.0}, mesh);
  get<::Tags::Modal<VectorTag<dim, 0>>>(package_data_up_xi.modal_volume_data)
      .get(0) = ModalVector{7.0, 5.0, 2.5, 0.0};
  get<::Tags::Modal<VectorTag<dim, 0>>>(package_data_lo_xi.modal_volume_data)
      .get(0) = ModalVector{-2.1, 1.1, 1.0e-18, 0.0};
  to_nodal_coefficients(&nodal_vector_expected.get(0),
                        {3.0, 2.0, 0.9 * 0.99, 0.5 * 0.99}, mesh);

  krivodonova(&nodal_scalar_data_to_limit, &nodal_vector_data_to_limit, element,
              mesh, neighbor_data);
  CHECK_ITERABLE_APPROX(get(nodal_scalar_data_to_limit),
                        get(nodal_scalar_expected));
  for (size_t i = 0; i < dim; ++i) {
    CAPTURE(i);
    CHECK_ITERABLE_APPROX(nodal_vector_data_to_limit.get(i),
                          nodal_vector_expected.get(i));
  }
}

void run() noexcept {
  INFO("Testing 1d limiter");
  for (size_t order = Spectral::minimum_number_of_points<
           Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto>;
       order < Spectral::maximum_number_of_points<Spectral::Basis::Legendre>;
       ++order) {
    test_package_data(order);
  }
  test_limiting_two_neighbors();
  test_limiting_different_values_different_tensors();
}
}  // namespace test_1d

namespace test_2d {
void test_package_data() noexcept {
  INFO("Testing package data");
  constexpr size_t dim = 2;
  const Mesh<dim> mesh({{2, 3}},
                       {{Spectral::Basis::Legendre, Spectral::Basis::Legendre}},
                       {{Spectral::Quadrature::GaussLobatto,
                         Spectral::Quadrature::GaussLobatto}});
  CAPTURE(mesh);
  using Limiter = Krivodonova<dim, tmpl::list<ScalarTag<0>, VectorTag<dim, 0>>>;
  Limiter krivodonova{};

  const ModalVector neighbor_modes{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  Scalar<DataVector> tensor0(mesh.number_of_grid_points());
  to_nodal_coefficients(&get(tensor0), neighbor_modes, mesh);

  tnsr::I<DataVector, dim> tensor1(mesh.number_of_grid_points());
  for (size_t d = 0; d < dim; ++d) {
    to_nodal_coefficients(&tensor1.get(d), (d + 2.0) * neighbor_modes, mesh);
  }
  Limiter::PackagedData packaged_data{};

  // test no reorienting
  {
    krivodonova.package_data(make_not_null(&packaged_data), tensor0, tensor1,
                             mesh, {});
    CHECK(get(get<::Tags::Modal<ScalarTag<0>>>(
              packaged_data.modal_volume_data)) == neighbor_modes);
    for (size_t d = 0; d < dim; ++d) {
      CHECK(
          get<::Tags::Modal<VectorTag<dim, 0>>>(packaged_data.modal_volume_data)
              .get(d) == ModalVector((d + 2.0) * neighbor_modes));
    }
  }
  // test reorienting
  {
    krivodonova.package_data(
        make_not_null(&packaged_data), tensor0, tensor1, mesh,
        OrientationMap<2>{
            {{Direction<2>::lower_eta(), Direction<2>::upper_xi()}}});
    const ModalVector expected_modes{2.0, 4.0, 6.0, 1.0, 3.0, 5.0};
    CHECK(get(get<::Tags::Modal<ScalarTag<0>>>(
              packaged_data.modal_volume_data)) == expected_modes);
    for (size_t d = 0; d < dim; ++d) {
      CHECK(
          get<::Tags::Modal<VectorTag<dim, 0>>>(packaged_data.modal_volume_data)
              .get(d) == ModalVector((d + 2.0) * expected_modes));
    }
  }
}

template <typename F>
void test_limiting_2_2_coefficient(const F& helper) noexcept {
  // Limit no coefficients because the highest coefficient doesn't need
  // limiting.
  helper({0.0, 1.0, 2.0, 3.0, 4.0, 7.0, 3.0, 4.1, 0.0},
         {0.0, 1.0, 2.0, 3.0, 4.0, 7.0, 3.0, 4.1, 0.0},
         {0.0, 1.0, 2.0, 3.0, 4.0, 3.0, -2.0, 2.0, 1.0},
         {0.0, 1.0, 2.0, 3.0, 4.0, 1.0, -5.0, 1.0e-18, 0.0},
         {0.0, 1.0, 2.0, 3.0, 4.0, 1.0, -5.0, 1.0e-18, 0.0},
         {0.0, 1.0, 2.0, 3.0, 4.0, 3.0, -2.0, 2.0, 1.0});

  // Limit (2,2) because +(1,2)
  helper({0.0, 1.0, 3.0, 3.0, 3.0, 7.0, 3.0, 2.5, 0.0},
         {0.0, 1.0, 3.0, 3.0, 3.0, 7.0, 3.0, 4.1, 0.0},
         {0.0, 1.0, -2.0, -2.0, -2.0, 2.0, -2.0, 2.0, 1.0},
         {0.0, 1.0, -5.0, -5.0, -5.0, 1.0, -5.0, 1.0e-18, 0.0},
         {0.0, 1.0, -5.0, -5.0, -5.0, 1.0, -5.0, 1.0e-18, 0.0},
         {0.0, 1.0, -2.0, -2.0, -2.0, 2.0, -2.0, 2.0, 0.5 * 0.99});

  // Limit (2,2) because +(2,1)
  helper({0.0, 1.0, 3.0, 3.0, 3.0, 4.1, 3.0, 4.1, 0.0},
         {0.0, 1.0, 3.0, 3.0, 3.0, 2.5, 3.0, 4.1, 0.0},
         {0.0, 1.0, -2.0, -2.0, -2.0, 2.0, -2.0, 2.0, 1.0},
         {0.0, 1.0, -5.0, -5.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0, 1.0, -5.0, -5.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0, 1.0, -2.0, -2.0, -2.0, 2.0, -2.0, 2.0, 0.5 * 0.99});

  // Limit (2,2) because -(1,2)
  helper({0.0, 1.0, 3.0, 3.0, 3.0, 4.1, 3.0, 4.1, 0.0},
         {0.0, 1.0, 3.0, 3.0, 3.0, 4.1, 3.0, 4.1, 0.0},
         {0.0, 1.0, -2.0, -2.0, -2.0, 2.0, -2.0, 2.0, 1.0},
         {0.0, 1.0, -5.0, -5.0, -5.0, 1.0e-18, -5.0, 1.3, 0.0},
         {0.0, 1.0, -5.0, -5.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0, 1.0, -2.0, -2.0, -2.0, 2.0, -2.0, 2.0, 0.7 * 0.99});

  // Limit (2,2) because -(2,1)
  helper({0.0, 1.0, 3.0, 3.0, 3.0, 4.1, 3.0, 4.1, 0.0},
         {0.0, 1.0, 3.0, 3.0, 3.0, 4.1, 3.0, 4.1, 0.0},
         {0.0, 1.0, -2.0, -2.0, -2.0, 2.0, -2.0, 2.0, 1.0},
         {0.0, 1.0, -5.0, -5.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0, 1.0, -5.0, -5.0, -5.0, 1.3, -5.0, 1.0e-18, 0.0},
         {0.0, 1.0, -2.0, -2.0, -2.0, 2.0, -2.0, 2.0, 0.7 * 0.99});

  // Zero (2,2) because +(1,2)
  helper({0.0, 1.0, 3.0, 3.0, 3.0, 4.1, 3.0, 1.9, 0.0},
         {0.0, 1.0, 3.0, 3.0, 3.0, 4.1, 3.0, 4.1, 0.0},
         {0.0, 1.0, -2.0, -2.0, -2.0, 2.0, -2.0, 2.0, 1.0},
         {0.0, 1.0, -5.0, -5.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0, 1.0, -5.0, -5.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0, 1.0, -2.0, -2.0, -2.0, 2.0, -2.0, 2.0, 0.0});

  // Zero (2,2) because +(2,1)
  helper({0.0, 1.0, 3.0, 3.0, 3.0, 4.1, 3.0, 4.1, 0.0},
         {0.0, 1.0, 3.0, 3.0, 3.0, 1.9, 3.0, 4.1, 0.0},
         {0.0, 1.0, -2.0, -2.0, -2.0, 2.0, -2.0, 2.0, 1.0},
         {0.0, 1.0, -5.0, -5.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0, 1.0, -5.0, -5.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0, 1.0, -2.0, -2.0, -2.0, 2.0, -2.0, 2.0, 0.0});

  // Zero (2,2) because -(1,2)
  helper({0.0, 1.0, 3.0, 3.0, 3.0, 4.1, 3.0, 4.1, 0.0},
         {0.0, 1.0, 3.0, 3.0, 3.0, 4.1, 3.0, 4.1, 0.0},
         {0.0, 1.0, -2.0, -2.0, -2.0, 2.0, -2.0, 2.0, 1.0},
         {0.0, 1.0, -5.0, -5.0, -5.0, 1.0e-18, -5.0, 5.3, 0.0},
         {0.0, 1.0, -5.0, -5.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0, 1.0, -2.0, -2.0, -2.0, 2.0, -2.0, 2.0, 0.0});

  // Zero (2,2) because -(2,1)
  helper({0.0, 1.0, 3.0, 3.0, 3.0, 4.1, 3.0, 4.1, 0.0},
         {0.0, 1.0, 3.0, 3.0, 3.0, 4.1, 3.0, 4.1, 0.0},
         {0.0, 1.0, -2.0, -2.0, -2.0, 2.0, -2.0, 2.0, 1.0},
         {0.0, 1.0, -5.0, -5.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0, 1.0, -5.0, -5.0, -5.0, 5.3, -5.0, 1.0e-18, 0.0},
         {0.0, 1.0, -2.0, -2.0, -2.0, 2.0, -2.0, 2.0, 0.0});
}

template <typename F>
void test_limiting_2_1_coefficient(const F& helper) noexcept {
  // Limit (2,1) because +(2,0)
  helper({0.0, 7.0, 3.0, 7.0, 3.0, 4.1, 7.0, 4.1, 0.0},
         {0.0, 7.0, 3.0, 7.0, 3.0, 4.1, 7.0, 4.1, 0.0},
         {0.0, 3.0, 2.0, 3.0, -2.0, 2.0, 3.0, 2.0, 1.0},
         {0.0, -2.1, -5.0, -2.1, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0, -2.1, -5.0, -2.1, -5.0, 5.3, -5.0, 1.0e-18, 0.0},
         {0.0, 3.0, 2.0, 3.0, -2.0, 0.99, 3.0, 2.0, 0.0});

  // Limit (2,1) because -(2,0)
  helper({0.0, 7.0, 5.0, 7.0, 3.0, 4.1, 7.0, 4.1, 0.0},
         {0.0, 7.0, 5.0, 7.0, 3.0, 4.1, 7.0, 4.1, 0.0},
         {0.0, 3.0, 2.0, 3.0, -2.0, 2.0, 3.0, 2.0, 1.0},
         {0.0, -2.1, 1.1, -2.1, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0, -2.1, 1.1, -2.1, -5.0, 5.3, -5.0, 1.0e-18, 0.0},
         {0.0, 3.0, 2.0, 3.0, -2.0, 0.9 * 0.99, 3.0, 2.0, 0.0});

  // Zero (2,1) because +(2,0)
  helper({0.0, 7.0, 1.0, 7.0, 3.0, 4.1, 7.0, 4.1, 0.0},
         {0.0, 7.0, 1.0, 7.0, 3.0, 4.1, 7.0, 4.1, 0.0},
         {0.0, 3.0, 2.0, 3.0, -2.0, 2.0, 3.0, 2.0, 1.0},
         {0.0, -2.1, 1.1, -2.1, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0, -2.1, 1.1, -2.1, -5.0, 5.3, -5.0, 1.0e-18, 0.0},
         {0.0, 3.0, 2.0, 3.0, -2.0, 0.0, 3.0, 2.0, 0.0});

  // Zero (2,1) because -(2,0)
  helper({0.0, 7.0, 5.0, 7.0, 3.0, 4.1, 7.0, 4.1, 0.0},
         {0.0, 7.0, 5.0, 7.0, 3.0, 4.1, 7.0, 4.1, 0.0},
         {0.0, 3.0, 2.0, 3.0, -2.0, 2.0, 3.0, 2.0, 1.0},
         {0.0, -2.1, 2.1, -2.1, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0, -2.1, 2.1, -2.1, -5.0, 5.3, -5.0, 1.0e-18, 0.0},
         {0.0, 3.0, 2.0, 3.0, -2.0, 0.0, 3.0, 2.0, 0.0});

  // Limit (2,1) because +(1,1)
  helper({0.0, 7.0, 5.0, 7.0, 0.0, 4.1, 7.0, 4.1, 0.0},
         {0.0, 7.0, 5.0, 7.0, 3.0, 4.1, 7.0, 4.1, 0.0},
         {0.0, 3.0, 2.0, 3.0, -2.0, 2.0, 3.0, 2.0, 1.0},
         {0.0, -2.1, -5.0, -2.1, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0, -2.1, -5.0, -2.1, -5.0, 5.3, -5.0, 1.0e-18, 0.0},
         {0.0, 3.0, 2.0, 3.0, -2.0, 0.99 * 2.0, 3.0, 2.0, 0.0});

  // Limit (2,1) because -(1,1)
  helper({0.0, 7.0, 5.0, 7.0, 3.0, 4.1, 7.0, 4.1, 0.0},
         {0.0, 7.0, 5.0, 7.0, 3.0, 4.1, 7.0, 4.1, 0.0},
         {0.0, 3.0, 2.0, 3.0, -2.0, 2.0, 3.0, 2.0, 1.0},
         {0.0, -2.1, -5.0, -2.1, -2.1, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0, -2.1, -5.0, -2.1, -5.0, 5.3, -5.0, 1.0e-18, 0.0},
         {0.0, 3.0, 2.0, 3.0, -2.0, 0.99 * 0.1, 3.0, 2.0, 0.0});

  // Zero (2,1) because +(1,1)
  helper({0.0, 7.0, 5.0, 7.0, -3.0, 4.1, 7.0, 4.1, 0.0},
         {0.0, 7.0, 5.0, 7.0, 3.0, 4.1, 7.0, 4.1, 0.0},
         {0.0, 3.0, 2.0, 3.0, -2.0, 2.0, 3.0, 2.0, 1.0},
         {0.0, -2.1, -5.0, -2.1, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0, -2.1, -5.0, -2.1, -5.0, 5.3, -5.0, 1.0e-18, 0.0},
         {0.0, 3.0, 2.0, 3.0, -2.0, 0.0, 3.0, 2.0, 0.0});

  // Zero (2,1) because -(1,1)
  helper({0.0, 7.0, 5.0, 7.0, 3.0, 4.1, 7.0, 4.1, 0.0},
         {0.0, 7.0, 5.0, 7.0, 3.0, 4.1, 7.0, 4.1, 0.0},
         {0.0, 3.0, 2.0, 3.0, -2.0, 2.0, 3.0, 2.0, 1.0},
         {0.0, -2.1, -5.0, -2.1, 5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0, -2.1, -5.0, -2.1, -5.0, 5.3, -5.0, 1.0e-18, 0.0},
         {0.0, 3.0, 2.0, 3.0, -2.0, 0.0, 3.0, 2.0, 0.0});
}

template <typename F>
void test_limiting_1_2_coefficient(const F& helper) noexcept {
  // Limit (1,2) because +(0,2)
  helper({0.0, 7.0, 7.0, 7.0, 3.0, 4.1, 4.0, 4.1, 0.0},
         {0.0, 7.0, 7.0, 7.0, 3.0, 4.1, 4.0, 4.1, 0.0},
         {0.0, 3.0, 2.0, 3.0, -2.0, 2.0, 3.0, 2.0, 1.0},
         {0.0, -2.1, -5.0, -2.1, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0, -2.1, -5.0, -2.1, -5.0, 5.3, -5.0, 1.0e-18, 0.0},
         {0.0, 3.0, 2.0, 3.0, -2.0, 2.0, 3.0, 0.99, 0.0});

  // Limit (1,2) because -(0,2)
  helper({0.0, 7.0, 7.0, 7.0, 3.0, 4.1, 7.0, 4.1, 0.0},
         {0.0, 7.0, 7.0, 7.0, 3.0, 4.1, 7.0, 4.1, 0.0},
         {0.0, 3.0, 2.0, 3.0, -2.0, 2.0, 3.0, 2.0, 1.0},
         {0.0, -2.1, -5.0, -2.1, -5.0, 1.0e-18, 2.1, 1.0e-18, 0.0},
         {0.0, -2.1, -5.0, -2.1, -5.0, 5.3, 2.1, 1.0e-18, 0.0},
         {0.0, 3.0, 2.0, 3.0, -2.0, 2.0, 3.0, 0.9 * 0.99, 0.0});

  // Zero (1,2) because +(0,2)
  helper({0.0, 7.0, 7.0, 7.0, 3.0, 4.1, 2.9, 4.1, 0.0},
         {0.0, 7.0, 7.0, 7.0, 3.0, 4.1, 2.9, 4.1, 0.0},
         {0.0, 3.0, 2.0, 3.0, -2.0, 2.0, 3.0, 2.0, 1.0},
         {0.0, -2.1, -5.0, -2.1, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0, -2.1, -5.0, -2.1, -5.0, 5.3, -5.0, 1.0e-18, 0.0},
         {0.0, 3.0, 2.0, 3.0, -2.0, 2.0, 3.0, 0.0, 0.0});

  // Zero (1,2) because -(0,2)
  helper({0.0, 7.0, 7.0, 7.0, 3.0, 4.1, 7.0, 4.1, 0.0},
         {0.0, 7.0, 7.0, 7.0, 3.0, 4.1, 7.0, 4.1, 0.0},
         {0.0, 3.0, 2.0, 3.0, -2.0, 2.0, 3.0, 2.0, 1.0},
         {0.0, -2.1, -5.0, -2.1, -5.0, 1.0e-18, 8.1, 1.0e-18, 0.0},
         {0.0, -2.1, -5.0, -2.1, -5.0, 5.3, 8.1, 1.0e-18, 0.0},
         {0.0, 3.0, 2.0, 3.0, -2.0, 2.0, 3.0, 0.0, 0.0});

  // Limit (2,1) because +(1,1)
  helper({0.0, 7.0, 5.0, 7.0, 3.0, 4.1, 7.0, 4.1, 0.0},
         {0.0, 7.0, 5.0, 7.0, 0.0, 4.1, 7.0, 4.1, 0.0},
         {0.0, 3.0, 2.0, 3.0, -2.0, 2.0, 3.0, 2.0, 1.0},
         {0.0, -2.1, -5.0, -2.1, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0, -2.1, -5.0, -2.1, -5.0, 5.3, -5.0, 1.0e-18, 0.0},
         {0.0, 3.0, 2.0, 3.0, -2.0, 2.0, 3.0, 0.99 * 2.0, 0.0});

  // Limit (2,1) because -(1,1)
  helper({0.0, 7.0, 5.0, 7.0, 3.0, 4.1, 7.0, 4.1, 0.0},
         {0.0, 7.0, 5.0, 7.0, 3.0, 4.1, 7.0, 4.1, 0.0},
         {0.0, 3.0, 2.0, 3.0, -2.0, 2.0, 3.0, 2.0, 1.0},
         {0.0, -2.1, -5.0, -2.1, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0, -2.1, -5.0, -2.1, -4.0, 5.3, -5.0, 1.0e-18, 0.0},
         {0.0, 3.0, 2.0, 3.0, -2.0, 2.0, 3.0, 0.99 * 2.0, 0.0});

  // Zero (2,1) because +(1,1)
  helper({0.0, 7.0, 5.0, 7.0, 3.0, 4.1, 7.0, 4.1, 0.0},
         {0.0, 7.0, 5.0, 7.0, -3.0, 4.1, 7.0, 4.1, 0.0},
         {0.0, 3.0, 2.0, 3.0, -2.0, 2.0, 3.0, 2.0, 1.0},
         {0.0, -2.1, -5.0, -2.1, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0, -2.1, -5.0, -2.1, -5.0, 5.3, -5.0, 1.0e-18, 0.0},
         {0.0, 3.0, 2.0, 3.0, -2.0, 2.0, 3.0, 0.0, 0.0});

  // Zero (2,1) because -(1,1)
  helper({0.0, 7.0, 5.0, 7.0, 3.0, 4.1, 7.0, 4.1, 0.0},
         {0.0, 7.0, 5.0, 7.0, 3.0, 4.1, 7.0, 4.1, 0.0},
         {0.0, 3.0, 2.0, 3.0, -2.0, 2.0, 3.0, 2.0, 1.0},
         {0.0, -2.1, -5.0, -2.1, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0, -2.1, -5.0, -2.1, -1.0, 5.3, -5.0, 1.0e-18, 0.0},
         {0.0, 3.0, 2.0, 3.0, -2.0, 2.0, 3.0, 0.0, 0.0});
}

template <typename F>
void test_limiting_2_0_coefficient(const F& helper) noexcept {
  // Limit (2,0) because +(1,0)
  helper({5.0, 3.1, 7.0, 7.0, 3.0, 4.1, 4.0, 4.1, 0.0},
         {5.0, 7.0, 7.0, 7.0, 3.0, 4.1, 4.0, 4.1, 0.0},
         {1.0, 3.0, 2.0, 3.0, 0.2, 2.0, 3.0, 2.0, 1.0},
         {-5.0, -2.1, -5.0, -2.1, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {-5.0, -2.1, -5.0, -2.1, -5.0, 5.3, -5.0, 1.0e-18, 0.0},
         {1.0, 3.0, 0.1 * 0.99, 3.0, 0.2, 2.0, 3.0, 0.99, 0.0});

  // Limit (2,0) because -(1,0)
  helper({5.0, 7.0, 7.0, 7.0, 3.0, 4.1, 4.0, 4.1, 0.0},
         {5.0, 7.0, 7.0, 7.0, 3.0, 4.1, 4.0, 4.1, 0.0},
         {1.0, 3.0, 2.0, 3.0, 0.2, 2.0, 3.0, 2.0, 1.0},
         {-5.0, 2.1, -5.0, -2.1, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {-5.0, -2.1, -5.0, -2.1, -5.0, 5.3, -5.0, 1.0e-18, 0.0},
         {1.0, 3.0, 0.9 * 0.99, 3.0, 0.2, 2.0, 3.0, 0.99, 0.0});

  // Zero (2,0) because +(1,0)
  helper({5.0, 2.9, 7.0, 7.0, 3.0, 4.1, 4.0, 4.1, 0.0},
         {5.0, 7.0, 7.0, 7.0, 3.0, 4.1, 4.0, 4.1, 0.0},
         {1.0, 3.0, 2.0, 3.0, 0.2, 2.0, 3.0, 2.0, 1.0},
         {-5.0, -2.1, -5.0, -2.1, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {-5.0, -2.1, -5.0, -2.1, -5.0, 5.3, -5.0, 1.0e-18, 0.0},
         {1.0, 3.0, 0.0, 3.0, 0.2, 2.0, 3.0, 0.99, 0.0});

  // Zero (2,0) because -(1,0)
  helper({5.0, 7.0, 7.0, 7.0, 3.0, 4.1, 4.0, 4.1, 0.0},
         {5.0, 7.0, 7.0, 7.0, 3.0, 4.1, 4.0, 4.1, 0.0},
         {1.0, 3.0, 2.0, 3.0, 0.2, 2.0, 3.0, 2.0, 1.0},
         {-5.0, 3.1, -5.0, -2.1, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {-5.0, -2.1, -5.0, -2.1, -5.0, 5.3, -5.0, 1.0e-18, 0.0},
         {1.0, 3.0, 0.0, 3.0, 0.2, 2.0, 3.0, 0.99, 0.0});
}

template <typename F>
void test_limiting_0_2_coefficient(const F& helper) noexcept {
  // Limit (0,2) because +(0,1)
  helper({5.0, 7.0, 7.0, 7.0, 3.0, 4.1, 4.0, 4.1, 0.0},
         {5.0, 7.0, 7.0, 3.1, 3.0, 4.1, 4.0, 4.1, 0.0},
         {1.0, 3.0, 2.0, 3.0, 0.2, 2.0, 3.0, 2.0, 1.0},
         {-5.0, -2.1, -5.0, -2.1, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {-5.0, -2.1, -5.0, -2.1, -5.0, 5.3, -5.0, 1.0e-18, 0.0},
         {1.0, 3.0, 2.0, 3.0, 0.2, 2.0, 0.1 * 0.99, 0.99, 0.0});

  // Limit (0,2) because -(0,1)
  helper({5.0, 7.0, 7.0, 7.0, 3.0, 4.1, 4.0, 4.1, 0.0},
         {5.0, 7.0, 7.0, 7.0, 3.0, 4.1, 4.0, 4.1, 0.0},
         {1.0, 3.0, 2.0, 3.0, 0.2, 2.0, 3.0, 2.0, 1.0},
         {-5.0, -2.1, -5.0, -2.1, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {-5.0, -2.1, -5.0, 2.2, -5.0, 5.3, -5.0, 1.0e-18, 0.0},
         {1.0, 3.0, 2.0, 3.0, 0.2, 2.0, 0.8 * 0.99, 0.99, 0.0});

  // Zero (0,2) because +(0,1)
  helper({5.0, 7.0, 7.0, 7.0, 3.0, 4.1, 4.0, 4.1, 0.0},
         {5.0, 7.0, 7.0, 2.9, 3.0, 4.1, 4.0, 4.1, 0.0},
         {1.0, 3.0, 2.0, 3.0, 0.2, 2.0, 3.0, 2.0, 1.0},
         {-5.0, -2.1, -5.0, -2.1, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {-5.0, -2.1, -5.0, -2.1, -5.0, 5.3, -5.0, 1.0e-18, 0.0},
         {1.0, 3.0, 2.0, 3.0, 0.2, 2.0, 0.0, 0.99, 0.0});

  // Zero (0,2) because -(0,1)
  helper({5.0, 7.0, 7.0, 7.0, 3.0, 4.1, 4.0, 4.1, 0.0},
         {5.0, 7.0, 7.0, 7.0, 3.0, 4.1, 4.0, 4.1, 0.0},
         {1.0, 3.0, 2.0, 3.0, 0.2, 2.0, 3.0, 2.0, 1.0},
         {-5.0, -2.1, -5.0, -2.1, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {-5.0, -2.1, -5.0, 3.2, -5.0, 5.3, -5.0, 1.0e-18, 0.0},
         {1.0, 3.0, 2.0, 3.0, 0.2, 2.0, 0.0, 0.99, 0.0});
}

template <typename F>
void test_limiting_1_1_coefficient(const F& helper) noexcept {
  // Limit (1,1) because +(1,0)
  helper({5.0, 4.0, 7.0, 7.0, 3.0, 4.1, 4.0, 4.1, 0.0},
         {5.0, 3.3, 7.0, 3.1, 3.0, 4.1, 4.0, 4.1, 0.0},
         {1.0, 3.0, 2.0, 3.0, 1.0, 2.0, 3.0, 2.0, 1.0},
         {-5.0, -2.1, -5.0, -2.1, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {-5.0, -2.1, -5.0, -2.1, -5.0, 5.3, -5.0, 1.0e-18, 0.0},
         {1.0, 3.0, 1.0 * 0.99, 3.0, 0.3 * 0.99, 2.0 * 0.99, 0.1 * 0.99, 0.99,
          0.0});

  // Limit (1,1) because -(1,0)
  helper({5.0, 4.0, 7.0, 7.0, 3.0, 4.1, 4.0, 4.1, 0.0},
         {5.0, 4.3, 7.0, 3.1, 3.0, 4.1, 4.0, 4.1, 0.0},
         {1.0, 3.0, 2.0, 3.0, 1.0, 2.0, 3.0, 2.0, 1.0},
         {-5.0, -2.1, -5.0, -2.1, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {-5.0, 2.6, -5.0, -2.1, -5.0, 5.3, -5.0, 1.0e-18, 0.0},
         {1.0, 3.0, 1.0 * 0.99, 3.0, 0.4 * 0.99, 2.0 * 0.99, 0.1 * 0.99, 0.99,
          0.0});

  // Limit (1,1) because +(0,1)
  helper({5.0, 3.3, 7.0, 3.4, 3.0, 4.1, 4.0, 4.1, 0.0},
         {5.0, 4.2, 7.0, 3.1, 3.0, 4.1, 4.0, 4.1, 0.0},
         {1.0, 3.0, 2.0, 3.0, 1.0, 2.0, 3.0, 2.0, 1.0},
         {-5.0, -2.1, -5.0, -2.1, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {-5.0, -2.1, -5.0, -2.1, -5.0, 5.3, -5.0, 1.0e-18, 0.0},
         {1.0, 3.0, 0.3 * 0.99, 3.0, 0.4 * 0.99, 2.0 * 0.99, 0.1 * 0.99, 0.99,
          0.0});

  // Limit (1,1) because -(0,1)
  helper({5.0, 4.0, 7.0, 7.0, 3.0, 4.1, 4.0, 4.1, 0.0},
         {5.0, 4.3, 7.0, 3.1, 3.0, 4.1, 4.0, 4.1, 0.0},
         {1.0, 3.0, 2.0, 3.0, 1.0, 2.0, 3.0, 2.0, 1.0},
         {-5.0, -2.1, -5.0, 2.5, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {-5.0, -2.1, -5.0, -2.1, -5.0, 5.3, -5.0, 1.0e-18, 0.0},
         {1.0, 3.0, 1.0 * 0.99, 3.0, 0.5 * 0.99, 2.0 * 0.99, 0.1 * 0.99, 0.99,
          0.0});

  // Zero (1,1) because +(1,0)
  helper({5.0, 4.0, 7.0, 7.0, 3.0, 4.1, 4.0, 4.1, 0.0},
         {5.0, 2.9, 7.0, 3.1, 3.0, 4.1, 4.0, 4.1, 0.0},
         {1.0, 3.0, 2.0, 3.0, 1.0, 2.0, 3.0, 2.0, 1.0},
         {-5.0, -2.1, -5.0, -2.1, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {-5.0, -2.1, -5.0, -2.1, -5.0, 5.3, -5.0, 1.0e-18, 0.0},
         {1.0, 3.0, 1.0 * 0.99, 3.0, 0.0, 2.0 * 0.99, 0.1 * 0.99, 0.99, 0.0});

  // Zero (1,1) because -(1,0)
  helper({5.0, 4.0, 7.0, 7.0, 3.0, 4.1, 4.0, 4.1, 0.0},
         {5.0, 4.3, 7.0, 3.1, 3.0, 4.1, 4.0, 4.1, 0.0},
         {1.0, 3.0, 2.0, 3.0, 1.0, 2.0, 3.0, 2.0, 1.0},
         {-5.0, -2.1, -5.0, -2.1, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {-5.0, 3.1, -5.0, -2.1, -5.0, 5.3, -5.0, 1.0e-18, 0.0},
         {1.0, 3.0, 1.0 * 0.99, 3.0, 0.0, 2.0 * 0.99, 0.1 * 0.99, 0.99, 0.0});

  // Zero (1,1) because +(0,1)
  helper({5.0, 3.3, 7.0, 2.9, 3.0, 4.1, 4.0, 4.1, 0.0},
         {5.0, 4.2, 7.0, 3.1, 3.0, 4.1, 4.0, 4.1, 0.0},
         {1.0, 3.0, 2.0, 3.0, 1.0, 2.0, 3.0, 2.0, 1.0},
         {-5.0, -2.1, -5.0, -2.1, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {-5.0, -2.1, -5.0, -2.1, -5.0, 5.3, -5.0, 1.0e-18, 0.0},
         {1.0, 3.0, 0.3 * 0.99, 3.0, 0.0, 2.0 * 0.99, 0.1 * 0.99, 0.99, 0.0});

  // Zero (1,1) because -(0,1)
  helper({5.0, 4.0, 7.0, 7.0, 3.0, 4.1, 4.0, 4.1, 0.0},
         {5.0, 4.3, 7.0, 3.1, 3.0, 4.1, 4.0, 4.1, 0.0},
         {1.0, 3.0, 2.0, 3.0, 1.0, 2.0, 3.0, 2.0, 1.0},
         {-5.0, -2.1, -5.0, 4.5, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {-5.0, -2.1, -5.0, -2.1, -5.0, 5.3, -5.0, 1.0e-18, 0.0},
         {1.0, 3.0, 1.0 * 0.99, 3.0, 0.0, 2.0 * 0.99, 0.1 * 0.99, 0.99, 0.0});
}

template <typename F>
void test_limiting_1_0_coefficient(const F& helper) noexcept {
  // Limit (1,0) because +(0,0)
  helper({1.5, 4.0, 7.0, 7.0, 3.0, 4.1, 4.0, 4.1, 0.0},
         {5.0, 3.3, 7.0, 3.1, 3.0, 4.1, 4.0, 4.1, 0.0},
         {1.0, 3.0, 2.0, 3.0, 1.0, 2.0, 3.0, 2.0, 1.0},
         {-5.0, -2.1, -5.0, -2.1, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {-5.0, -2.1, -5.0, -2.1, -5.0, 5.3, -5.0, 1.0e-18, 0.0},
         {1.0, 0.5 * 0.99, 1.0 * 0.99, 3.0, 0.3 * 0.99, 2.0 * 0.99, 0.1 * 0.99,
          0.99, 0.0});

  // Limit (1,0) because -(0,0)
  helper({5.0, 4.0, 7.0, 7.0, 3.0, 4.1, 4.0, 4.1, 0.0},
         {5.0, 3.3, 7.0, 3.1, 3.0, 4.1, 4.0, 4.1, 0.0},
         {1.0, 3.0, 2.0, 3.0, 1.0, 2.0, 3.0, 2.0, 1.0},
         {0.6, -2.1, -5.0, -2.1, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {-5.0, -2.1, -5.0, -2.1, -5.0, 5.3, -5.0, 1.0e-18, 0.0},
         {1.0, 0.4 * 0.99, 1.0 * 0.99, 3.0, 0.3 * 0.99, 2.0 * 0.99, 0.1 * 0.99,
          0.99, 0.0});

  // Zero (1,0) because +(0,0)
  helper({0.5, 4.0, 7.0, 7.0, 3.0, 4.1, 4.0, 4.1, 0.0},
         {5.0, 3.3, 7.0, 3.1, 3.0, 4.1, 4.0, 4.1, 0.0},
         {1.0, 3.0, 2.0, 3.0, 1.0, 2.0, 3.0, 2.0, 1.0},
         {-5.0, -2.1, -5.0, -2.1, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {-5.0, -2.1, -5.0, -2.1, -5.0, 5.3, -5.0, 1.0e-18, 0.0},
         {1.0, 0.0, 1.0 * 0.99, 3.0, 0.3 * 0.99, 2.0 * 0.99, 0.1 * 0.99, 0.99,
          0.0});

  // Zero (1,0) because -(0,0)
  helper({5.0, 4.0, 7.0, 7.0, 3.0, 4.1, 4.0, 4.1, 0.0},
         {5.0, 3.3, 7.0, 3.1, 3.0, 4.1, 4.0, 4.1, 0.0},
         {1.0, 3.0, 2.0, 3.0, 1.0, 2.0, 3.0, 2.0, 1.0},
         {1.6, -2.1, -5.0, -2.1, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {-5.0, -2.1, -5.0, -2.1, -5.0, 5.3, -5.0, 1.0e-18, 0.0},
         {1.0, 0.0, 1.0 * 0.99, 3.0, 0.3 * 0.99, 2.0 * 0.99, 0.1 * 0.99, 0.99,
          0.0});
}

template <typename F>
void test_limiting_0_1_coefficient(const F& helper) noexcept {
  // Limit (0,1) because +(0,0)
  helper({1.4, 4.0, 7.0, 7.0, 3.0, 4.1, 4.0, 4.1, 0.0},
         {1.5, 3.3, 7.0, 3.1, 3.0, 4.1, 4.0, 4.1, 0.0},
         {1.0, 3.0, 2.0, 3.0, 1.0, 2.0, 3.0, 2.0, 1.0},
         {-5.0, -2.1, -5.0, -2.1, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {-5.0, -2.1, -5.0, -2.1, -5.0, 5.3, -5.0, 1.0e-18, 0.0},
         {1.0, 0.4 * 0.99, 1.0 * 0.99, 0.5 * 0.99, 0.3 * 0.99, 2.0 * 0.99,
          0.1 * 0.99, 0.99, 0.0});

  // Limit (0,1) because -(0,0)
  helper({0.9, 4.0, 7.0, 7.0, 3.0, 4.1, 4.0, 4.1, 0.0},
         {5.0, 3.3, 7.0, 3.1, 3.0, 4.1, 4.0, 4.1, 0.0},
         {1.0, 3.0, 2.0, 3.0, 1.0, 2.0, 3.0, 2.0, 1.0},
         {-5.0, -2.1, -5.0, -2.1, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.7, -2.1, -5.0, -2.1, -5.0, 5.3, -5.0, 1.0e-18, 0.0},
         {1.0, 0.0, 1.0 * 0.99, 0.3 * 0.99, 0.3 * 0.99, 2.0 * 0.99, 0.1 * 0.99,
          0.99, 0.0});

  // Zero (0,1) because +(0,0)
  helper({1.4, 4.0, 7.0, 7.0, 3.0, 4.1, 4.0, 4.1, 0.0},
         {0.5, 3.3, 7.0, 3.1, 3.0, 4.1, 4.0, 4.1, 0.0},
         {1.0, 3.0, 2.0, 3.0, 1.0, 2.0, 3.0, 2.0, 1.0},
         {-5.0, -2.1, -5.0, -2.1, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {-5.0, -2.1, -5.0, -2.1, -5.0, 5.3, -5.0, 1.0e-18, 0.0},
         {1.0, 0.4 * 0.99, 1.0 * 0.99, 0.0, 0.3 * 0.99, 2.0 * 0.99, 0.1 * 0.99,
          0.99, 0.0});

  // Zero (0,1) because -(0,0)
  helper({5.0, 4.0, 7.0, 7.0, 3.0, 4.1, 4.0, 4.1, 0.0},
         {5.0, 3.3, 7.0, 3.1, 3.0, 4.1, 4.0, 4.1, 0.0},
         {1.0, 3.0, 2.0, 3.0, 1.0, 2.0, 3.0, 2.0, 1.0},
         {0.4, -2.1, -5.0, -2.1, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {1.7, -2.1, -5.0, -2.1, -5.0, 5.3, -5.0, 1.0e-18, 0.0},
         {1.0, 0.6 * 0.99, 1.0 * 0.99, 0.0, 0.3 * 0.99, 2.0 * 0.99, 0.1 * 0.99,
          0.99, 0.0});
}

void test_limiting_different_values_different_tensors() noexcept {
  INFO("Testing different values for each tensor component");
  constexpr size_t dim = 2;
  const size_t order = 2;  // Use only 3 coefficients because more is tedious...
  const Mesh<dim> mesh(order + 1, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto);
  const size_t num_pts = mesh.number_of_grid_points();
  using Limiter = Krivodonova<dim, tmpl::list<ScalarTag<0>, VectorTag<dim, 0>>>;
  // Use non-unity (because that's the default) but close alpha values to make
  // the math easier but still test thoroughly.
  Limiter krivodonova{
      make_array<Spectral::maximum_number_of_points<Spectral::Basis::Legendre>>(
          0.99)};

  NeighborData<dim, typename Limiter::PackagedData> neighbor_data{};

  const Element<dim> element(ElementId<dim>{0}, {});
  // We don't care about the ElementId for these tests, just the direction.
  Limiter::PackagedData& package_data_up_xi = neighbor_data[std::make_pair(
      Direction<dim>::upper_xi(), ElementId<dim>{0})];
  Limiter::PackagedData& package_data_lo_xi = neighbor_data[std::make_pair(
      Direction<dim>::lower_xi(), ElementId<dim>{0})];
  Limiter::PackagedData& package_data_up_eta = neighbor_data[std::make_pair(
      Direction<dim>::upper_eta(), ElementId<dim>{0})];
  Limiter::PackagedData& package_data_lo_eta = neighbor_data[std::make_pair(
      Direction<dim>::lower_eta(), ElementId<dim>{0})];

  package_data_up_xi.modal_volume_data.initialize(num_pts);
  package_data_up_xi.mesh = mesh;
  package_data_lo_xi.modal_volume_data.initialize(num_pts);
  package_data_lo_xi.mesh = mesh;
  package_data_up_eta.modal_volume_data.initialize(num_pts);
  package_data_up_eta.mesh = mesh;
  package_data_lo_eta.modal_volume_data.initialize(num_pts);
  package_data_lo_eta.mesh = mesh;

  Scalar<DataVector> nodal_scalar_data_to_limit(num_pts, 0.0);
  tnsr::I<DataVector, dim> nodal_vector_data_to_limit(num_pts, 0.0);
  Scalar<DataVector> nodal_scalar_expected(num_pts, 0.0);
  tnsr::I<DataVector, dim> nodal_vector_expected(num_pts, 0.0);

  // Limit (1,0) because +(0,0)
  to_nodal_coefficients(&get(nodal_scalar_data_to_limit),
                        {1.0, 3.0, 2.0, 3.0, 1.0, 2.0, 3.0, 2.0, 1.0}, mesh);
  get(get<::Tags::Modal<ScalarTag<0>>>(package_data_up_xi.modal_volume_data)) =
      ModalVector{1.5, 4.0, 7.0, 7.0, 3.0, 4.1, 4.0, 4.1, 0.0};
  get(get<::Tags::Modal<ScalarTag<0>>>(package_data_lo_xi.modal_volume_data)) =
      ModalVector{-5.0, -2.1, -5.0, -2.1, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0};
  get(get<::Tags::Modal<ScalarTag<0>>>(package_data_up_eta.modal_volume_data)) =
      ModalVector{5.0, 3.3, 7.0, 3.1, 3.0, 4.1, 4.0, 4.1, 0.0};
  get(get<::Tags::Modal<ScalarTag<0>>>(package_data_lo_eta.modal_volume_data)) =
      ModalVector{-5.0, -2.1, -5.0, -2.1, -5.0, 5.3, -5.0, 1.0e-18, 0.0};
  to_nodal_coefficients(&get(nodal_scalar_expected),
                        {1.0, 0.5 * 0.99, 1.0 * 0.99, 3.0, 0.3 * 0.99,
                         2.0 * 0.99, 0.1 * 0.99, 0.99, 0.0},
                        mesh);

  // Zero (1,2) because +(0,2)
  to_nodal_coefficients(&nodal_vector_data_to_limit.get(0),
                        {0.0, 3.0, 2.0, 3.0, -2.0, 2.0, 3.0, 2.0, 1.0}, mesh);
  get<::Tags::Modal<VectorTag<dim, 0>>>(package_data_up_xi.modal_volume_data)
      .get(0) = ModalVector{0.0, 7.0, 7.0, 7.0, 3.0, 4.1, 2.9, 4.1, 0.0};
  get<::Tags::Modal<VectorTag<dim, 0>>>(package_data_lo_xi.modal_volume_data)
      .get(0) =
      ModalVector{0.0, -2.1, -5.0, -2.1, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0};
  get<::Tags::Modal<VectorTag<dim, 0>>>(package_data_up_eta.modal_volume_data)
      .get(0) = ModalVector{0.0, 7.0, 7.0, 7.0, 3.0, 4.1, 2.9, 4.1, 0.0};
  get<::Tags::Modal<VectorTag<dim, 0>>>(package_data_lo_eta.modal_volume_data)
      .get(0) =
      ModalVector{0.0, -2.1, -5.0, -2.1, -5.0, 5.3, -5.0, 1.0e-18, 0.0};
  to_nodal_coefficients(&nodal_vector_expected.get(0),
                        {0.0, 3.0, 2.0, 3.0, -2.0, 2.0, 3.0, 0.0, 0.0}, mesh);

  // Limit (2,1) because +(1,1)
  to_nodal_coefficients(&nodal_vector_data_to_limit.get(1),
                        {0.0, 3.0, 2.0, 3.0, -2.0, 2.0, 3.0, 2.0, 1.0}, mesh);
  get<::Tags::Modal<VectorTag<dim, 0>>>(package_data_up_xi.modal_volume_data)
      .get(1) = ModalVector{0.0, 7.0, 5.0, 7.0, 3.0, 4.1, 7.0, 4.1, 0.0};
  get<::Tags::Modal<VectorTag<dim, 0>>>(package_data_lo_xi.modal_volume_data)
      .get(1) =
      ModalVector{0.0, -2.1, -5.0, -2.1, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0};
  get<::Tags::Modal<VectorTag<dim, 0>>>(package_data_up_eta.modal_volume_data)
      .get(1) = ModalVector{0.0, 7.0, 5.0, 7.0, 3.0, 4.1, 7.0, 4.1, 0.0};
  get<::Tags::Modal<VectorTag<dim, 0>>>(package_data_lo_eta.modal_volume_data)
      .get(1) =
      ModalVector{0.0, -2.1, -5.0, -2.1, -4.0, 5.3, -5.0, 1.0e-18, 0.0};
  to_nodal_coefficients(&nodal_vector_expected.get(1),
                        {0.0, 3.0, 2.0, 3.0, -2.0, 2.0, 3.0, 0.99 * 2.0, 0.0},
                        mesh);

  krivodonova(&nodal_scalar_data_to_limit, &nodal_vector_data_to_limit, element,
              mesh, neighbor_data);
  CHECK_ITERABLE_APPROX(get(nodal_scalar_data_to_limit),
                        get(nodal_scalar_expected));
  for (size_t i = 0; i < dim; ++i) {
    CAPTURE(i);
    CHECK_ITERABLE_APPROX(nodal_vector_data_to_limit.get(i),
                          nodal_vector_expected.get(i));
  }
}

void run() noexcept {
  INFO("Testing 2d limiter");
  test_package_data();

  INFO("Testing applying limiter to coefficients");
  constexpr size_t dim = 2;
  const size_t order = 2;  // Use only 3 coefficients because more is tedious...
  const Mesh<dim> mesh(order + 1, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto);
  const size_t num_pts = mesh.number_of_grid_points();
  using Limiter = Krivodonova<dim, tmpl::list<ScalarTag<0>>>;
  // Use non-unity (because that's the default) but close alpha values to make
  // the math easier but still test thoroughly.
  Limiter krivodonova{
      make_array<Spectral::maximum_number_of_points<Spectral::Basis::Legendre>>(
          0.99)};

  NeighborData<dim, typename Limiter::PackagedData> neighbor_data{};

  const Element<dim> element(ElementId<dim>{0}, {});
  // We don't care about the ElementId for these tests, just the direction.
  Limiter::PackagedData& package_data_up_xi = neighbor_data[std::make_pair(
      Direction<dim>::upper_xi(), ElementId<dim>{0})];
  Limiter::PackagedData& package_data_lo_xi = neighbor_data[std::make_pair(
      Direction<dim>::lower_xi(), ElementId<dim>{0})];
  Limiter::PackagedData& package_data_up_eta = neighbor_data[std::make_pair(
      Direction<dim>::upper_eta(), ElementId<dim>{0})];
  Limiter::PackagedData& package_data_lo_eta = neighbor_data[std::make_pair(
      Direction<dim>::lower_eta(), ElementId<dim>{0})];

  package_data_up_xi.modal_volume_data.initialize(num_pts);
  package_data_up_xi.mesh = mesh;
  package_data_lo_xi.modal_volume_data.initialize(num_pts);
  package_data_lo_xi.mesh = mesh;
  package_data_up_eta.modal_volume_data.initialize(num_pts);
  package_data_up_eta.mesh = mesh;
  package_data_lo_eta.modal_volume_data.initialize(num_pts);
  package_data_lo_eta.mesh = mesh;

  Scalar<DataVector> nodal_scalar_data_to_limit(num_pts, 0.0);
  DataVector expected(num_pts);
  const auto helper =
      [
        &element, &expected, &krivodonova, &mesh, &neighbor_data,
        &package_data_lo_xi, &package_data_up_xi, &package_data_lo_eta,
        &package_data_up_eta, &nodal_scalar_data_to_limit
      ](const ModalVector& up_xi_coeffs, const ModalVector& up_eta_coeffs,
        const ModalVector& initial_coeffs, const ModalVector& lo_xi_coeffs,
        const ModalVector& lo_eta_coeffs,
        const ModalVector& expected_coeffs) noexcept {
    to_nodal_coefficients(&get(nodal_scalar_data_to_limit), initial_coeffs,
                          mesh);
    get(get<::Tags::Modal<ScalarTag<0>>>(
        package_data_up_xi.modal_volume_data)) = up_xi_coeffs;
    get(get<::Tags::Modal<ScalarTag<0>>>(
        package_data_lo_xi.modal_volume_data)) = lo_xi_coeffs;
    get(get<::Tags::Modal<ScalarTag<0>>>(
        package_data_up_eta.modal_volume_data)) = up_eta_coeffs;
    get(get<::Tags::Modal<ScalarTag<0>>>(
        package_data_lo_eta.modal_volume_data)) = lo_eta_coeffs;
    krivodonova(&nodal_scalar_data_to_limit, element, mesh, neighbor_data);
    to_nodal_coefficients(&expected, expected_coeffs, mesh);
    CHECK_ITERABLE_APPROX(get(nodal_scalar_data_to_limit), expected);
  };

  // Map between 2D and 1D coefficients:
  // [(0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (0,2), (1,2), (2,2)]
  //  (0,     1,     2,     3,     4,     5,     6,     7,     8)

  test_limiting_2_2_coefficient(helper);
  test_limiting_2_1_coefficient(helper);
  test_limiting_1_2_coefficient(helper);
  test_limiting_2_0_coefficient(helper);
  test_limiting_0_2_coefficient(helper);
  test_limiting_1_1_coefficient(helper);
  test_limiting_1_0_coefficient(helper);
  test_limiting_0_1_coefficient(helper);

  test_limiting_different_values_different_tensors();
}
}  // namespace test_2d

namespace test_3d {
void test_package_data() noexcept {
  INFO("Testing package data");
  constexpr size_t dim = 3;
  const Mesh<dim> mesh(
      {{2, 3, 4}},
      {{Spectral::Basis::Legendre, Spectral::Basis::Legendre,
        Spectral::Basis::Legendre}},
      {{Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::GaussLobatto,
        Spectral::Quadrature::GaussLobatto}});
  CAPTURE(mesh);
  const auto logical_x = logical_coordinates(mesh);
  using Limiter = Krivodonova<dim, tmpl::list<ScalarTag<0>, VectorTag<dim, 0>>>;
  Limiter krivodonova{};

  const ModalVector neighbor_modes{
      0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0,
      12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0};
  Scalar<DataVector> tensor0(mesh.number_of_grid_points());
  to_nodal_coefficients(&get(tensor0), neighbor_modes, mesh);

  tnsr::I<DataVector, dim> tensor1(mesh.number_of_grid_points());
  for (size_t d = 0; d < dim; ++d) {
    to_nodal_coefficients(&tensor1.get(d), (d + 2.0) * neighbor_modes, mesh);
  }
  Limiter::PackagedData packaged_data{};

  // test no reorienting
  {
    krivodonova.package_data(make_not_null(&packaged_data), tensor0, tensor1,
                             mesh, {});
    CHECK(get(get<::Tags::Modal<ScalarTag<0>>>(
              packaged_data.modal_volume_data)) == neighbor_modes);
    for (size_t d = 0; d < dim; ++d) {
      CHECK(
          get<::Tags::Modal<VectorTag<dim, 0>>>(packaged_data.modal_volume_data)
              .get(d) == ModalVector((d + 2.0) * neighbor_modes));
    }
  }
  // test reorienting
  {
    krivodonova.package_data(make_not_null(&packaged_data), tensor0, tensor1,
                             mesh,
                             OrientationMap<3>{{{Direction<3>::upper_zeta(),
                                                 Direction<3>::upper_eta(),
                                                 Direction<3>::upper_xi()}}});
    const ModalVector expected_modes{
        0.0, 6.0, 12.0, 18.0, 2.0, 8.0, 14.0, 20.0, 4.0, 10.0, 16.0, 22.0,
        1.0, 7.0, 13.0, 19.0, 3.0, 9.0, 15.0, 21.0, 5.0, 11.0, 17.0, 23.0};
    CHECK(get(get<::Tags::Modal<ScalarTag<0>>>(
              packaged_data.modal_volume_data)) == expected_modes);
    for (size_t d = 0; d < dim; ++d) {
      CHECK(
          get<::Tags::Modal<VectorTag<dim, 0>>>(packaged_data.modal_volume_data)
              .get(d) == ModalVector((d + 2.0) * expected_modes));
    }
  }
}

template <typename F>
void test_limiting_2_2_2_coefficient(const F& helper) noexcept {
  // Limit no coefficients because the highest coefficient doesn't need
  // limiting.
  helper({0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
          9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 4.1,
          18.0, 19.0, 20.0, 21.0, 22.0, 4.1,  3.0,  4.1,  0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
          9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 4.1,
          18.0, 19.0, 20.0, 21.0, 22.0, 4.1,  3.0,  4.1,  0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
          9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 4.1,
          18.0, 19.0, 20.0, 21.0, 22.0, 4.1,  3.0,  4.1,  0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
          9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 2.0,
          18.0, 19.0, 20.0, 21.0, 22.0, 3.0,  -2.0, 2.0,  1.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     8.0,
          9.0,  10.0, 11.0, 12.0, 13.0, 14.0,    15.0, 16.0,    1.0e-18,
          18.0, 19.0, 20.0, 21.0, 22.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     8.0,
          9.0,  10.0, 11.0, 12.0, 13.0, 14.0,    15.0, 16.0,    1.0e-18,
          18.0, 19.0, 20.0, 21.0, 22.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     8.0,
          9.0,  10.0, 11.0, 12.0, 13.0, 14.0,    15.0, 16.0,    1.0e-18,
          18.0, 19.0, 20.0, 21.0, 22.0, 1.0e-18, -5.0, 1.0e-18, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
          9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 2.0,
          18.0, 19.0, 20.0, 21.0, 22.0, 3.0,  -2.0, 2.0,  1.0});

  {  // Limit (2,2,2)
    // Limit (2,2,2) because +(1,2,2)
    helper({0.0,  1.0,  2.0,  3.0,  4.0,  5.0, 6.0,  7.0, 3.0,
            9.0,  10.0, 11.0, 12.0, 13.0, 3.0, 15.0, 3.0, 4.1,
            18.0, 19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  2.5, 0.0},
           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0, 6.0,  7.0, 3.0,
            9.0,  10.0, 11.0, 12.0, 13.0, 3.0, 15.0, 3.0, 4.1,
            18.0, 19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  4.1, 0.0},
           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0, 6.0,  7.0, 3.0,
            9.0,  10.0, 11.0, 12.0, 13.0, 3.0, 15.0, 3.0, 4.1,
            18.0, 19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  4.1, 0.0},

           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
            18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  1.0},

           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
            18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
            18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
            18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},

           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
            18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  0.5 * 0.99});

    // Limit (2,2,2) because +(2,1,2)
    helper({0.0,  1.0,  2.0,  3.0,  4.0,  5.0, 6.0,  7.0, 3.0,
            9.0,  10.0, 11.0, 12.0, 13.0, 3.0, 15.0, 3.0, 4.1,
            18.0, 19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  4.1, 0.0},
           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0, 6.0,  7.0, 3.0,
            9.0,  10.0, 11.0, 12.0, 13.0, 3.0, 15.0, 3.0, 4.1,
            18.0, 19.0, 3.0,  21.0, 3.0,  2.5, 3.0,  4.1, 0.0},
           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0, 6.0,  7.0, 3.0,
            9.0,  10.0, 11.0, 12.0, 13.0, 3.0, 15.0, 3.0, 4.1,
            18.0, 19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  4.1, 0.0},

           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
            18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  1.0},

           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
            18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
            18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
            18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},

           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
            18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  0.5 * 0.99});

    // Limit (2,2,2) because +(2,2,1)
    helper({0.0,  1.0,  2.0,  3.0,  4.0,  5.0, 6.0,  7.0, 3.0,
            9.0,  10.0, 11.0, 12.0, 13.0, 3.0, 15.0, 3.0, 4.1,
            18.0, 19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  4.1, 0.0},
           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0, 6.0,  7.0, 3.0,
            9.0,  10.0, 11.0, 12.0, 13.0, 3.0, 15.0, 3.0, 4.1,
            18.0, 19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  4.1, 0.0},
           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0, 6.0,  7.0, 3.0,
            9.0,  10.0, 11.0, 12.0, 13.0, 3.0, 15.0, 3.0, 2.5,
            18.0, 19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  4.1, 0.0},

           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
            18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  1.0},

           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
            18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
            18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
            18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},

           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
            18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  0.5 * 0.99});

    // Limit (2,2,2) because -(1,2,2)
    helper({0.0,  1.0,  2.0,  3.0,  4.0,  5.0, 6.0,  7.0, 3.0,
            9.0,  10.0, 11.0, 12.0, 13.0, 3.0, 15.0, 3.0, 4.1,
            18.0, 19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  4.1, 0.0},
           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0, 6.0,  7.0, 3.0,
            9.0,  10.0, 11.0, 12.0, 13.0, 3.0, 15.0, 3.0, 4.1,
            18.0, 19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  4.1, 0.0},
           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0, 6.0,  7.0, 3.0,
            9.0,  10.0, 11.0, 12.0, 13.0, 3.0, 15.0, 3.0, 4.1,
            18.0, 19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  4.1, 0.0},

           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
            18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  1.0},

           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,  -5.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0, 1.0e-18,
            18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.3,  0.0},
           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
            18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
            18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},

           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
            18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  0.7 * 0.99});

    // Limit (2,2,2) because -(2,1,2)
    helper({0.0,  1.0,  2.0,  3.0,  4.0,  5.0, 6.0,  7.0, 3.0,
            9.0,  10.0, 11.0, 12.0, 13.0, 3.0, 15.0, 3.0, 4.1,
            18.0, 19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  4.1, 0.0},
           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0, 6.0,  7.0, 3.0,
            9.0,  10.0, 11.0, 12.0, 13.0, 3.0, 15.0, 3.0, 4.1,
            18.0, 19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  4.1, 0.0},
           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0, 6.0,  7.0, 3.0,
            9.0,  10.0, 11.0, 12.0, 13.0, 3.0, 15.0, 3.0, 4.1,
            18.0, 19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  4.1, 0.0},

           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
            18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  1.0},

           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
            18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,     -5.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -5.0, 15.0, -5.0,    1.0e-18,
            18.0, 19.0, -5.0, 21.0, -5.0, 1.3,  -5.0, 1.0e-18, 0.0},
           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
            18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},

           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
            18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  0.7 * 0.99});

    // Limit (2,2,2) because -(2,2,1)
    helper({0.0,  1.0,  2.0,  3.0,  4.0,  5.0, 6.0,  7.0, 3.0,
            9.0,  10.0, 11.0, 12.0, 13.0, 3.0, 15.0, 3.0, 4.1,
            18.0, 19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  4.1, 0.0},
           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0, 6.0,  7.0, 3.0,
            9.0,  10.0, 11.0, 12.0, 13.0, 3.0, 15.0, 3.0, 4.1,
            18.0, 19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  4.1, 0.0},
           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0, 6.0,  7.0, 3.0,
            9.0,  10.0, 11.0, 12.0, 13.0, 3.0, 15.0, 3.0, 4.1,
            18.0, 19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  4.1, 0.0},

           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
            18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  1.0},

           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
            18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
            18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.3,
            18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},

           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
            18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  0.7 * 0.99});
  }

  {  // Zero (2,2,2)
    // Zero (2,2,2) because +(1,2,2)
    helper({0.0,  1.0,  2.0,  3.0,  4.0,  5.0, 6.0,  7.0, 3.0,
            9.0,  10.0, 11.0, 12.0, 13.0, 3.0, 15.0, 3.0, 4.1,
            18.0, 19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  1.9, 0.0},
           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0, 6.0,  7.0, 3.0,
            9.0,  10.0, 11.0, 12.0, 13.0, 3.0, 15.0, 3.0, 4.1,
            18.0, 19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  4.1, 0.0},
           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0, 6.0,  7.0, 3.0,
            9.0,  10.0, 11.0, 12.0, 13.0, 3.0, 15.0, 3.0, 4.1,
            18.0, 19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  4.1, 0.0},

           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
            18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  1.0},

           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
            18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
            18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
            18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},

           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
            18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  0.0});

    // Zero (2,2,2) because +(2,1,2)
    helper({0.0,  1.0,  2.0,  3.0,  4.0,  5.0, 6.0,  7.0, 3.0,
            9.0,  10.0, 11.0, 12.0, 13.0, 3.0, 15.0, 3.0, 4.1,
            18.0, 19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  4.1, 0.0},
           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0, 6.0,  7.0, 3.0,
            9.0,  10.0, 11.0, 12.0, 13.0, 3.0, 15.0, 3.0, 4.1,
            18.0, 19.0, 3.0,  21.0, 3.0,  1.9, 3.0,  4.1, 0.0},
           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0, 6.0,  7.0, 3.0,
            9.0,  10.0, 11.0, 12.0, 13.0, 3.0, 15.0, 3.0, 4.1,
            18.0, 19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  4.1, 0.0},

           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
            18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  1.0},

           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
            18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
            18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
            18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},

           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
            18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  0.0});

    // Zero (2,2,2) because +(2,2,1)
    helper({0.0,  1.0,  2.0,  3.0,  4.0,  5.0, 6.0,  7.0, 3.0,
            9.0,  10.0, 11.0, 12.0, 13.0, 3.0, 15.0, 3.0, 4.1,
            18.0, 19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  4.1, 0.0},
           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0, 6.0,  7.0, 3.0,
            9.0,  10.0, 11.0, 12.0, 13.0, 3.0, 15.0, 3.0, 4.1,
            18.0, 19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  4.1, 0.0},
           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0, 6.0,  7.0, 3.0,
            9.0,  10.0, 11.0, 12.0, 13.0, 3.0, 15.0, 3.0, 1.9,
            18.0, 19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  4.1, 0.0},

           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
            18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  1.0},

           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
            18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
            18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
            18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},

           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
            18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  0.0});

    // Zero (2,2,2) because -(1,2,2)
    helper({0.0,  1.0,  2.0,  3.0,  4.0,  5.0, 6.0,  7.0, 3.0,
            9.0,  10.0, 11.0, 12.0, 13.0, 3.0, 15.0, 3.0, 4.1,
            18.0, 19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  4.1, 0.0},
           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0, 6.0,  7.0, 3.0,
            9.0,  10.0, 11.0, 12.0, 13.0, 3.0, 15.0, 3.0, 4.1,
            18.0, 19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  4.1, 0.0},
           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0, 6.0,  7.0, 3.0,
            9.0,  10.0, 11.0, 12.0, 13.0, 3.0, 15.0, 3.0, 4.1,
            18.0, 19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  4.1, 0.0},

           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
            18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  1.0},

           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,  -5.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0, 1.0e-18,
            18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 5.3,  0.0},
           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
            18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
            18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},

           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
            18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  0.0});

    // Zero (2,2,2) because -(2,1,2)
    helper({0.0,  1.0,  2.0,  3.0,  4.0,  5.0, 6.0,  7.0, 3.0,
            9.0,  10.0, 11.0, 12.0, 13.0, 3.0, 15.0, 3.0, 4.1,
            18.0, 19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  4.1, 0.0},
           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0, 6.0,  7.0, 3.0,
            9.0,  10.0, 11.0, 12.0, 13.0, 3.0, 15.0, 3.0, 4.1,
            18.0, 19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  4.1, 0.0},
           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0, 6.0,  7.0, 3.0,
            9.0,  10.0, 11.0, 12.0, 13.0, 3.0, 15.0, 3.0, 4.1,
            18.0, 19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  4.1, 0.0},

           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
            18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  1.0},

           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
            18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,     -5.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -5.0, 15.0, -5.0,    1.0e-18,
            18.0, 19.0, -5.0, 21.0, -5.0, 5.3,  -5.0, 1.0e-18, 0.0},
           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
            18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},

           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
            18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  0.0});

    // Zero (2,2,2) because -(1,2,2)
    helper({0.0,  1.0,  2.0,  3.0,  4.0,  5.0, 6.0,  7.0, 3.0,
            9.0,  10.0, 11.0, 12.0, 13.0, 3.0, 15.0, 3.0, 4.1,
            18.0, 19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  4.1, 0.0},
           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0, 6.0,  7.0, 3.0,
            9.0,  10.0, 11.0, 12.0, 13.0, 3.0, 15.0, 3.0, 4.1,
            18.0, 19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  4.1, 0.0},
           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0, 6.0,  7.0, 3.0,
            9.0,  10.0, 11.0, 12.0, 13.0, 3.0, 15.0, 3.0, 4.1,
            18.0, 19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  4.1, 0.0},

           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
            18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  1.0},

           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
            18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
            18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    5.3,
            18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},

           {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
            9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
            18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  0.0});
  }
}

template <typename F>
void test_limiting_2_2_1_coefficient(const F& helper) noexcept {
  // Limit (2,2,1) because +(1,2,1)
  helper({0.0,  1.0,   2.0,  3.0,  4.0,  5.0, 6.0,  3.0,  3.0,
          9.0,  10.0,  11.0, 12.0, 13.0, 2.0, 15.0, -1.1, 4.1,
          18.0, -19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  2.5,  0.0},
    {0.0,  1.0,  2.0,  3.0,   4.0,  2.0, 6.0,  7.0, 3.0,
          9.0,  10.0, 11.0, 12.0,  13.0, 3.0, 15.0, 3.0, 4.1,
          18.0, 19.0, 3.0,  -21.0, 3.0,  4.1, 3.0,  4.1, 0.0},
    {0.0,  1.0,  2.0,   3.0,  4.0,  5.0, 6.0,   7.0, 3.0,
          9.0,  10.0, -11.0, 12.0, 13.0, 3.0, -15.0, 3.0, 4.1,
          18.0, 19.0, 3.0,   21.0, 3.0,  4.1, 3.0,   4.1, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  1.0},

    {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  11.0,    -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -8.0,    1.0e-18,
          18.0, 49.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
    {0.0,  1.0,  2.0,  3.0,  4.0,  8.0,     6.0,  7.0,     -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
          18.0, 19.0, -5.0, 43.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
    {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
          9.0,  10.0, 28.0, 12.0, 13.0, -5.0,    31.0, -5.0,    1.0e-18,
          18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},

    {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 0.9 * 0.99,
          18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  0.5 * 0.99});

  // Zero (2,2,1) because +(1,2,1)
  helper({0.0,  1.0,   2.0,  3.0,  4.0,  5.0, 6.0,  3.0,   3.0,
          9.0,  10.0,  11.0, 12.0, 13.0, 2.0, 15.0, -16.0, 4.1,
          18.0, -19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  2.5,   0.0},
         {0.0,  1.0,  2.0,  3.0,   4.0,  2.0, 6.0,  7.0, 3.0,
          9.0,  10.0, 11.0, 12.0,  13.0, 3.0, 15.0, 3.0, 4.1,
          18.0, 19.0, 3.0,  -21.0, 3.0,  4.1, 3.0,  4.1, 0.0},
         {0.0,  1.0,  2.0,   3.0,  4.0,  5.0, 6.0,   7.0, 3.0,
          9.0,  10.0, -11.0, 12.0, 13.0, 3.0, -15.0, 3.0, 4.1,
          18.0, 19.0, 3.0,   21.0, 3.0,  4.1, 3.0,   4.1, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  1.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  11.0,    -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -8.0,    1.0e-18,
          18.0, 49.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  8.0,     6.0,  7.0,     -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
          18.0, 19.0, -5.0, 43.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
          9.0,  10.0, 28.0, 12.0, 13.0, -5.0,    31.0, -5.0,    1.0e-18,
          18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 0.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  0.5 * 0.99});

  // Limit (2,2,1) because -(1,2,1)
  helper({0.0,  1.0,   2.0,  3.0,  4.0,  5.0, 6.0,  3.0, 3.0,
          9.0,  10.0,  11.0, 12.0, 13.0, 2.0, 15.0, 1.0, 4.1,
          18.0, -19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  2.5, 0.0},
         {0.0,  1.0,  2.0,  3.0,   4.0,  2.0, 6.0,  7.0, 3.0,
          9.0,  10.0, 11.0, 12.0,  13.0, 3.0, 15.0, 3.0, 4.1,
          18.0, 19.0, 3.0,  -21.0, 3.0,  4.1, 3.0,  4.1, 0.0},
         {0.0,  1.0,  2.0,   3.0,  4.0,  5.0, 6.0,   7.0, 3.0,
          9.0,  10.0, -11.0, 12.0, 13.0, 3.0, -15.0, 3.0, 4.1,
          18.0, 19.0, 3.0,   21.0, 3.0,  4.1, 3.0,   4.1, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  1.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  11.0,    -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -3.0,    1.0e-18,
          18.0, 49.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  8.0,     6.0,  7.0,     -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
          18.0, 19.0, -5.0, 43.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
          9.0,  10.0, 28.0, 12.0, 13.0, -5.0,    31.0, -5.0,    1.0e-18,
          18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 1.0 * 0.99,
          18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  0.5 * 0.99});

  // Zero (2,2,1) because -(1,2,1)
  helper({0.0,  1.0,   2.0,  3.0,  4.0,  5.0, 6.0,  3.0, 3.0,
          9.0,  10.0,  11.0, 12.0, 13.0, 2.0, 15.0, 1.0, 4.1,
          18.0, -19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  2.5, 0.0},
         {0.0,  1.0,  2.0,  3.0,   4.0,  2.0, 6.0,  7.0, 3.0,
          9.0,  10.0, 11.0, 12.0,  13.0, 3.0, 15.0, 3.0, 4.1,
          18.0, 19.0, 3.0,  -21.0, 3.0,  4.1, 3.0,  4.1, 0.0},
         {0.0,  1.0,  2.0,   3.0,  4.0,  5.0, 6.0,   7.0, 3.0,
          9.0,  10.0, -11.0, 12.0, 13.0, 3.0, -15.0, 3.0, 4.1,
          18.0, 19.0, 3.0,   21.0, 3.0,  4.1, 3.0,   4.1, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  1.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  11.0,    -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, 16.0,    1.0e-18,
          18.0, 49.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  8.0,     6.0,  7.0,     -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
          18.0, 19.0, -5.0, 43.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
          9.0,  10.0, 28.0, 12.0, 13.0, -5.0,    31.0, -5.0,    1.0e-18,
          18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 0.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  0.5 * 0.99});

  // Limit (2,2,1) because +(2,1,1)
  helper({0.0,  1.0,   2.0,  3.0,  4.0,  5.0, 6.0,  3.0, 3.0,
          9.0,  10.0,  11.0, 12.0, 13.0, 2.0, 15.0, 1.0, 4.1,
          18.0, -19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  2.5, 0.0},
         {0.0,  1.0,  2.0,  3.0,   4.0,  2.0,  6.0,  7.0, 3.0,
          9.0,  10.0, 11.0, 12.0,  13.0, -1.3, 15.0, 3.0, 4.1,
          18.0, 19.0, 3.0,  -21.0, 3.0,  4.1,  3.0,  4.1, 0.0},
         {0.0,  1.0,  2.0,   3.0,  4.0,  5.0, 6.0,   7.0, 3.0,
          9.0,  10.0, -11.0, 12.0, 13.0, 3.0, -15.0, 3.0, 4.1,
          18.0, 19.0, 3.0,   21.0, 3.0,  4.1, 3.0,   4.1, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  1.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  11.0,    -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -8.0,    1.0e-18,
          18.0, 49.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  8.0,     6.0,  7.0,     -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
          18.0, 19.0, -5.0, 43.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
          9.0,  10.0, 28.0, 12.0, 13.0, -5.0,    31.0, -5.0,    1.0e-18,
          18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 0.7 * 0.99,
          18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  0.5 * 0.99});

  // Zero (2,2,1) because +(2,1,1)
  helper({0.0,  1.0,   2.0,  3.0,  4.0,  5.0, 6.0,  3.0, 3.0,
          9.0,  10.0,  11.0, 12.0, 13.0, 2.0, 15.0, 1.0, 4.1,
          18.0, -19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  2.5, 0.0},
         {0.0,  1.0,  2.0,  3.0,   4.0,  2.0,   6.0,  7.0, 3.0,
          9.0,  10.0, 11.0, 12.0,  13.0, -14.0, 15.0, 3.0, 4.1,
          18.0, 19.0, 3.0,  -21.0, 3.0,  4.1,   3.0,  4.1, 0.0},
         {0.0,  1.0,  2.0,   3.0,  4.0,  5.0, 6.0,   7.0, 3.0,
          9.0,  10.0, -11.0, 12.0, 13.0, 3.0, -15.0, 3.0, 4.1,
          18.0, 19.0, 3.0,   21.0, 3.0,  4.1, 3.0,   4.1, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  1.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  11.0,    -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -8.0,    1.0e-18,
          18.0, 49.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  8.0,     6.0,  7.0,     -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
          18.0, 19.0, -5.0, 43.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
          9.0,  10.0, 28.0, 12.0, 13.0, -5.0,    31.0, -5.0,    1.0e-18,
          18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 0.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  0.5 * 0.99});

  // Limit (2,2,1) because -(2,1,1)
  helper({0.0,  1.0,   2.0,  3.0,  4.0,  5.0, 6.0,  3.0, 3.0,
          9.0,  10.0,  11.0, 12.0, 13.0, 2.0, 15.0, 1.0, 4.1,
          18.0, -19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  2.5, 0.0},
         {0.0,  1.0,  2.0,  3.0,   4.0,  2.0, 6.0,  7.0, 3.0,
          9.0,  10.0, 11.0, 12.0,  13.0, 2.0, 15.0, 3.0, 4.1,
          18.0, 19.0, 3.0,  -21.0, 3.0,  4.1, 3.0,  4.1, 0.0},
         {0.0,  1.0,  2.0,   3.0,  4.0,  5.0, 6.0,   7.0, 3.0,
          9.0,  10.0, -11.0, 12.0, 13.0, 3.0, -15.0, 3.0, 4.1,
          18.0, 19.0, 3.0,   21.0, 3.0,  4.1, 3.0,   4.1, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  1.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  11.0,    -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -8.0,    1.0e-18,
          18.0, 49.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  8.0,     6.0,  7.0,     -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.2,    15.0, -5.0,    1.0e-18,
          18.0, 19.0, -5.0, 43.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
          9.0,  10.0, 28.0, 12.0, 13.0, -5.0,    31.0, -5.0,    1.0e-18,
          18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 0.2 * 0.99,
          18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  0.5 * 0.99});

  // Zero (2,2,1) because -(2,1,1)
  helper({0.0,  1.0,   2.0,  3.0,  4.0,  5.0, 6.0,  3.0, 3.0,
          9.0,  10.0,  11.0, 12.0, 13.0, 2.0, 15.0, 1.0, 4.1,
          18.0, -19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  2.5, 0.0},
         {0.0,  1.0,  2.0,  3.0,   4.0,  2.0, 6.0,  7.0, 3.0,
          9.0,  10.0, 11.0, 12.0,  13.0, 2.0, 15.0, 3.0, 4.1,
          18.0, 19.0, 3.0,  -21.0, 3.0,  4.1, 3.0,  4.1, 0.0},
         {0.0,  1.0,  2.0,   3.0,  4.0,  5.0, 6.0,   7.0, 3.0,
          9.0,  10.0, -11.0, 12.0, 13.0, 3.0, -15.0, 3.0, 4.1,
          18.0, 19.0, 3.0,   21.0, 3.0,  4.1, 3.0,   4.1, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  1.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  11.0,    -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -8.0,    1.0e-18,
          18.0, 49.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  8.0,     6.0,  7.0,     -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0, 12.0,    15.0, -5.0,    1.0e-18,
          18.0, 19.0, -5.0, 43.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
          9.0,  10.0, 28.0, 12.0, 13.0, -5.0,    31.0, -5.0,    1.0e-18,
          18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 0.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  0.5 * 0.99});

  // Limit (2,2,1) because +(2,2,0)
  helper({0.0,  1.0,   2.0,  3.0,  4.0,  5.0, 6.0,  3.0, 3.0,
          9.0,  10.0,  11.0, 12.0, 13.0, 2.0, 15.0, 1.0, 4.1,
          18.0, -19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  2.5, 0.0},
         {0.0,  1.0,  2.0,  3.0,   4.0,  2.0, 6.0,  7.0, 3.0,
          9.0,  10.0, 11.0, 12.0,  13.0, 2.0, 15.0, 3.0, 4.1,
          18.0, 19.0, 3.0,  -21.0, 3.0,  4.1, 3.0,  4.1, 0.0},
         {0.0,  1.0,  2.0,   3.0,  4.0,  5.0, 6.0,   7.0, -0.5,
          9.0,  10.0, -11.0, 12.0, 13.0, 3.0, -15.0, 3.0, 4.1,
          18.0, 19.0, 3.0,   21.0, 3.0,  4.1, 3.0,   4.1, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  1.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  11.0,    -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -8.0,    1.0e-18,
          18.0, 49.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  8.0,     6.0,  7.0,     -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
          18.0, 19.0, -5.0, 43.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
          9.0,  10.0, 28.0, 12.0, 13.0, -5.0,    31.0, -5.0,    1.0e-18,
          18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 1.5 * 0.99,
          18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  0.5 * 0.99});

  // Zero (2,2,1) because +(2,2,0)
  helper({0.0,  1.0,   2.0,  3.0,  4.0,  5.0, 6.0,  3.0, 3.0,
          9.0,  10.0,  11.0, 12.0, 13.0, 2.0, 15.0, 1.0, 4.1,
          18.0, -19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  2.5, 0.0},
         {0.0,  1.0,  2.0,  3.0,   4.0,  2.0, 6.0,  7.0, 3.0,
          9.0,  10.0, 11.0, 12.0,  13.0, 2.0, 15.0, 3.0, 4.1,
          18.0, 19.0, 3.0,  -21.0, 3.0,  4.1, 3.0,  4.1, 0.0},
         {0.0,  1.0,  2.0,   3.0,  4.0,  5.0, 6.0,   7.0, -8.0,
          9.0,  10.0, -11.0, 12.0, 13.0, 3.0, -15.0, 3.0, 4.1,
          18.0, 19.0, 3.0,   21.0, 3.0,  4.1, 3.0,   4.1, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  1.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  11.0,    -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -8.0,    1.0e-18,
          18.0, 49.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  8.0,     6.0,  7.0,     -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
          18.0, 19.0, -5.0, 43.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
          9.0,  10.0, 28.0, 12.0, 13.0, -5.0,    31.0, -5.0,    1.0e-18,
          18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 0.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  0.5 * 0.99});

  // Limit (2,2,1) because -(2,2,0)
  helper({0.0,  1.0,   2.0,  3.0,  4.0,  5.0, 6.0,  3.0, 3.0,
          9.0,  10.0,  11.0, 12.0, 13.0, 2.0, 15.0, 1.0, 4.1,
          18.0, -19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  2.5, 0.0},
         {0.0,  1.0,  2.0,  3.0,   4.0,  2.0, 6.0,  7.0, 3.0,
          9.0,  10.0, 11.0, 12.0,  13.0, 2.0, 15.0, 3.0, 4.1,
          18.0, 19.0, 3.0,  -21.0, 3.0,  4.1, 3.0,  4.1, 0.0},
         {0.0,  1.0,  2.0,   3.0,  4.0,  5.0, 6.0,   7.0, 3.0,
          9.0,  10.0, -11.0, 12.0, 13.0, 3.0, -15.0, 3.0, 4.1,
          18.0, 19.0, 3.0,   21.0, 3.0,  4.1, 3.0,   4.1, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  1.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  11.0,    -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -8.0,    1.0e-18,
          18.0, 49.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  8.0,     6.0,  7.0,     -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
          18.0, 19.0, -5.0, 43.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -3.8,
          9.0,  10.0, 28.0, 12.0, 13.0, -5.0,    31.0, -5.0,    1.0e-18,
          18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 1.8 * 0.99,
          18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  0.5 * 0.99});

  // Zero (2,2,1) because -(2,2,0)
  helper({0.0,  1.0,   2.0,  3.0,  4.0,  5.0, 6.0,  3.0, 3.0,
          9.0,  10.0,  11.0, 12.0, 13.0, 2.0, 15.0, 1.0, 4.1,
          18.0, -19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  2.5, 0.0},
         {0.0,  1.0,  2.0,  3.0,   4.0,  2.0, 6.0,  7.0, 3.0,
          9.0,  10.0, 11.0, 12.0,  13.0, 2.0, 15.0, 3.0, 4.1,
          18.0, 19.0, 3.0,  -21.0, 3.0,  4.1, 3.0,  4.1, 0.0},
         {0.0,  1.0,  2.0,   3.0,  4.0,  5.0, 6.0,   7.0, 3.0,
          9.0,  10.0, -11.0, 12.0, 13.0, 3.0, -15.0, 3.0, 4.1,
          18.0, 19.0, 3.0,   21.0, 3.0,  4.1, 3.0,   4.1, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  1.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  11.0,    -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -8.0,    1.0e-18,
          18.0, 49.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  8.0,     6.0,  7.0,     -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
          18.0, 19.0, -5.0, 43.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -0.8,
          9.0,  10.0, 28.0, 12.0, 13.0, -5.0,    31.0, -5.0,    1.0e-18,
          18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 0.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  0.5 * 0.99});
}

template <typename F>
void test_limiting_2_1_2_coefficient(const F& helper) noexcept {
  // Limit (2,1,2) because +(1,1,2)
  helper({0.0,  1.0,   2.0,  3.0,  4.0,   5.0, 6.0,  3.0, 3.0,
          9.0,  10.0,  11.0, 12.0, 13.0,  2.0, 15.0, 3.0, 4.1,
          18.0, -19.0, 3.0,  21.0, -1.15, 4.1, 3.0,  2.5, 0.0},
         {0.0,  1.0,  2.0,  3.0,   4.0,  2.0, 6.0,  7.0, 3.0,
          9.0,  10.0, 11.0, 12.0,  13.0, 3.0, 15.0, 3.0, 4.1,
          18.0, 19.0, 3.0,  -21.0, 3.0,  4.1, 3.0,  4.1, 0.0},
         {0.0,  1.0,  2.0,   3.0,  4.0,  5.0, 6.0,   7.0, 3.0,
          9.0,  10.0, -11.0, 12.0, 13.0, 3.0, -15.0, 3.0, 4.1,
          18.0, 19.0, 3.0,   21.0, 3.0,  4.1, 3.0,   4.1, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  1.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  11.0,    -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -8.0,    1.0e-18,
          18.0, 49.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  8.0,     6.0,  7.0,     -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
          18.0, 19.0, -5.0, 43.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
          9.0,  10.0, 28.0, 12.0, 13.0, -5.0,    31.0, -5.0,    1.0e-18,
          18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,         6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0,        15.0, -2.0, 2.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 0.85 * 0.99, -2.0, 2.0,  0.5 * 0.99});

  // Zero (2,1,2) because +(1,1,2)
  helper({0.0,  1.0,   2.0,  3.0,  4.0,   5.0, 6.0,  3.0, 3.0,
          9.0,  10.0,  11.0, 12.0, 13.0,  2.0, 15.0, 3.0, 4.1,
          18.0, -19.0, 3.0,  21.0, -10.0, 4.1, 3.0,  2.5, 0.0},
         {0.0,  1.0,  2.0,  3.0,   4.0,  2.0, 6.0,  7.0, 3.0,
          9.0,  10.0, 11.0, 12.0,  13.0, 3.0, 15.0, 3.0, 4.1,
          18.0, 19.0, 3.0,  -21.0, 3.0,  4.1, 3.0,  4.1, 0.0},
         {0.0,  1.0,  2.0,   3.0,  4.0,  5.0, 6.0,   7.0, 3.0,
          9.0,  10.0, -11.0, 12.0, 13.0, 3.0, -15.0, 3.0, 4.1,
          18.0, 19.0, 3.0,   21.0, 3.0,  4.1, 3.0,   4.1, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  1.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  11.0,    -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -8.0,    1.0e-18,
          18.0, 49.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  8.0,     6.0,  7.0,     -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
          18.0, 19.0, -5.0, 43.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
          9.0,  10.0, 28.0, 12.0, 13.0, -5.0,    31.0, -5.0,    1.0e-18,
          18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 0.0,  -2.0, 2.0,  0.5 * 0.99});

  // Limit (2,1,2) because -(1,1,2)
  helper({0.0,  1.0,   2.0,  3.0,  4.0,  5.0, 6.0,  3.0, 3.0,
          9.0,  10.0,  11.0, 12.0, 13.0, 2.0, 15.0, 3.0, 4.1,
          18.0, -19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  2.5, 0.0},
         {0.0,  1.0,  2.0,  3.0,   4.0,  2.0, 6.0,  7.0, 3.0,
          9.0,  10.0, 11.0, 12.0,  13.0, 3.0, 15.0, 3.0, 4.1,
          18.0, 19.0, 3.0,  -21.0, 3.0,  4.1, 3.0,  4.1, 0.0},
         {0.0,  1.0,  2.0,   3.0,  4.0,  5.0, 6.0,   7.0, 3.0,
          9.0,  10.0, -11.0, 12.0, 13.0, 3.0, -15.0, 3.0, 4.1,
          18.0, 19.0, 3.0,   21.0, 3.0,  4.1, 3.0,   4.1, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  1.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,   5.0,     6.0,  11.0,    -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0,  -5.0,    15.0, -8.0,    1.0e-18,
          18.0, 49.0, -5.0, 21.0, -3.05, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  8.0,     6.0,  7.0,     -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
          18.0, 19.0, -5.0, 43.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
          9.0,  10.0, 28.0, 12.0, 13.0, -5.0,    31.0, -5.0,    1.0e-18,
          18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,         6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0,        15.0, -2.0, 2.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 1.05 * 0.99, -2.0, 2.0,  0.5 * 0.99});

  // Zero (2,1,2) because -(1,1,2)
  helper({0.0,  1.0,   2.0,  3.0,  4.0,  5.0, 6.0,  3.0, 3.0,
          9.0,  10.0,  11.0, 12.0, 13.0, 2.0, 15.0, 3.0, 4.1,
          18.0, -19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  2.5, 0.0},
         {0.0,  1.0,  2.0,  3.0,   4.0,  2.0, 6.0,  7.0, 3.0,
          9.0,  10.0, 11.0, 12.0,  13.0, 3.0, 15.0, 3.0, 4.1,
          18.0, 19.0, 3.0,  -21.0, 3.0,  4.1, 3.0,  4.1, 0.0},
         {0.0,  1.0,  2.0,   3.0,  4.0,  5.0, 6.0,   7.0, 3.0,
          9.0,  10.0, -11.0, 12.0, 13.0, 3.0, -15.0, 3.0, 4.1,
          18.0, 19.0, 3.0,   21.0, 3.0,  4.1, 3.0,   4.1, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  1.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  11.0,    -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -8.0,    1.0e-18,
          18.0, 49.0, -5.0, 21.0, 22.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  8.0,     6.0,  7.0,     -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
          18.0, 19.0, -5.0, 43.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
          9.0,  10.0, 28.0, 12.0, 13.0, -5.0,    31.0, -5.0,    1.0e-18,
          18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 0.0,  -2.0, 2.0,  0.5 * 0.99});

  // Limit (2,1,2) because +(2,0,2)
  helper({0.0,  1.0,   2.0,  3.0,  4.0,  5.0, 6.0,  3.0, 3.0,
          9.0,  10.0,  11.0, 12.0, 13.0, 2.0, 15.0, 3.0, 4.1,
          18.0, -19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  2.5, 0.0},
         {0.0,  1.0,  2.0,   3.0,   4.0,  2.0, 6.0,  7.0, 3.0,
          9.0,  10.0, 11.0,  12.0,  13.0, 3.0, 15.0, 3.0, 4.1,
          18.0, 19.0, -1.15, -21.0, 3.0,  4.1, 3.0,  4.1, 0.0},
         {0.0,  1.0,  2.0,   3.0,  4.0,  5.0, 6.0,   7.0, 3.0,
          9.0,  10.0, -11.0, 12.0, 13.0, 3.0, -15.0, 3.0, 4.1,
          18.0, 19.0, 3.0,   21.0, 3.0,  4.1, 3.0,   4.1, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  1.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  11.0,    -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -8.0,    1.0e-18,
          18.0, 49.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  8.0,     6.0,  7.0,     -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
          18.0, 19.0, -5.0, 43.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
          9.0,  10.0, 28.0, 12.0, 13.0, -5.0,    31.0, -5.0,    1.0e-18,
          18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,         6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0,        15.0, -2.0, 2.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 0.85 * 0.99, -2.0, 2.0,  0.5 * 0.99});

  // Zero (2,1,2) because +(2,0,2)
  helper({0.0,  1.0,   2.0,  3.0,  4.0,  5.0, 6.0,  3.0, 3.0,
          9.0,  10.0,  11.0, 12.0, 13.0, 2.0, 15.0, 3.0, 4.1,
          18.0, -19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  2.5, 0.0},
         {0.0,  1.0,  2.0,  3.0,   4.0,  2.0, 6.0,  7.0, 3.0,
          9.0,  10.0, 11.0, 12.0,  13.0, 3.0, 15.0, 3.0, 4.1,
          18.0, 19.0, -4.0, -21.0, 3.0,  4.1, 3.0,  4.1, 0.0},
         {0.0,  1.0,  2.0,   3.0,  4.0,  5.0, 6.0,   7.0, 3.0,
          9.0,  10.0, -11.0, 12.0, 13.0, 3.0, -15.0, 3.0, 4.1,
          18.0, 19.0, 3.0,   21.0, 3.0,  4.1, 3.0,   4.1, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  1.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  11.0,    -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -8.0,    1.0e-18,
          18.0, 49.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  8.0,     6.0,  7.0,     -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
          18.0, 19.0, -5.0, 43.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
          9.0,  10.0, 28.0, 12.0, 13.0, -5.0,    31.0, -5.0,    1.0e-18,
          18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 0.0,  -2.0, 2.0,  0.5 * 0.99});

  // Limit (2,1,2) because -(2,0,2)
  helper({0.0,  1.0,   2.0,  3.0,  4.0,  5.0, 6.0,  3.0, 3.0,
          9.0,  10.0,  11.0, 12.0, 13.0, 2.0, 15.0, 3.0, 4.1,
          18.0, -19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  2.5, 0.0},
         {0.0,  1.0,  2.0,  3.0,   4.0,  2.0, 6.0,  7.0, 3.0,
          9.0,  10.0, 11.0, 12.0,  13.0, 3.0, 15.0, 3.0, 4.1,
          18.0, 19.0, 3.0,  -21.0, 3.0,  4.1, 3.0,  4.1, 0.0},
         {0.0,  1.0,  2.0,   3.0,  4.0,  5.0, 6.0,   7.0, 3.0,
          9.0,  10.0, -11.0, 12.0, 13.0, 3.0, -15.0, 3.0, 4.1,
          18.0, 19.0, 3.0,   21.0, 3.0,  4.1, 3.0,   4.1, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  1.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  11.0,    -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -8.0,    1.0e-18,
          18.0, 49.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,   3.0,  4.0,  8.0,     6.0,  7.0,     -5.0,
          9.0,  10.0, 11.0,  12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
          18.0, 19.0, -2.25, 43.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
          9.0,  10.0, 28.0, 12.0, 13.0, -5.0,    31.0, -5.0,    1.0e-18,
          18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,         6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0,        15.0, -2.0, 2.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 0.25 * 0.99, -2.0, 2.0,  0.5 * 0.99});

  // Zero (2,1,2) because -(2,0,2)
  helper({0.0,  1.0,   2.0,  3.0,  4.0,  5.0, 6.0,  3.0, 3.0,
          9.0,  10.0,  11.0, 12.0, 13.0, 2.0, 15.0, 3.0, 4.1,
          18.0, -19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  2.5, 0.0},
         {0.0,  1.0,  2.0,  3.0,   4.0,  2.0, 6.0,  7.0, 3.0,
          9.0,  10.0, 11.0, 12.0,  13.0, 3.0, 15.0, 3.0, 4.1,
          18.0, 19.0, 3.0,  -21.0, 3.0,  4.1, 3.0,  4.1, 0.0},
         {0.0,  1.0,  2.0,   3.0,  4.0,  5.0, 6.0,   7.0, 3.0,
          9.0,  10.0, -11.0, 12.0, 13.0, 3.0, -15.0, 3.0, 4.1,
          18.0, 19.0, 3.0,   21.0, 3.0,  4.1, 3.0,   4.1, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  1.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  11.0,    -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -8.0,    1.0e-18,
          18.0, 49.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  8.0,     6.0,  7.0,     -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
          18.0, 19.0, 7.0,  43.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
          9.0,  10.0, 28.0, 12.0, 13.0, -5.0,    31.0, -5.0,    1.0e-18,
          18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 0.0,  -2.0, 2.0,  0.5 * 0.99});

  // Limit (2,1,2) because +(2,1,1)
  helper({0.0,  1.0,   2.0,  3.0,  4.0,  5.0, 6.0,  3.0, 3.0,
          9.0,  10.0,  11.0, 12.0, 13.0, 2.0, 15.0, 3.0, 4.1,
          18.0, -19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  2.5, 0.0},
         {0.0,  1.0,  2.0,  3.0,   4.0,  2.0, 6.0,  7.0, 3.0,
          9.0,  10.0, 11.0, 12.0,  13.0, 3.0, 15.0, 3.0, 4.1,
          18.0, 19.0, 3.0,  -21.0, 3.0,  4.1, 3.0,  4.1, 0.0},
         {0.0,  1.0,  2.0,   3.0,  4.0,  5.0,   6.0,   7.0, 3.0,
          9.0,  10.0, -11.0, 12.0, 13.0, -1.33, -15.0, 3.0, 4.1,
          18.0, 19.0, 3.0,   21.0, 3.0,  4.1,   3.0,   4.1, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  1.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  11.0,    -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -8.0,    1.0e-18,
          18.0, 49.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  8.0,     6.0,  7.0,     -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
          18.0, 19.0, -5.0, 43.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
          9.0,  10.0, 28.0, 12.0, 13.0, -5.0,    31.0, -5.0,    1.0e-18,
          18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,         6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0,        15.0, -2.0, 2.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 0.67 * 0.99, -2.0, 2.0,  0.5 * 0.99});

  // Zero (2,1,2) because +(2,1,1)
  helper({0.0,  1.0,   2.0,  3.0,  4.0,  5.0, 6.0,  3.0, 3.0,
          9.0,  10.0,  11.0, 12.0, 13.0, 2.0, 15.0, 3.0, 4.1,
          18.0, -19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  2.5, 0.0},
         {0.0,  1.0,  2.0,  3.0,   4.0,  2.0, 6.0,  7.0, 3.0,
          9.0,  10.0, 11.0, 12.0,  13.0, 3.0, 15.0, 3.0, 4.1,
          18.0, 19.0, 3.0,  -21.0, 3.0,  4.1, 3.0,  4.1, 0.0},
         {0.0,  1.0,  2.0,   3.0,  4.0,  5.0,  6.0,   7.0, 3.0,
          9.0,  10.0, -11.0, 12.0, 13.0, -4.0, -15.0, 3.0, 4.1,
          18.0, 19.0, 3.0,   21.0, 3.0,  4.1,  3.0,   4.1, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  1.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  11.0,    -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -8.0,    1.0e-18,
          18.0, 49.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  8.0,     6.0,  7.0,     -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
          18.0, 19.0, -5.0, 43.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
          9.0,  10.0, 28.0, 12.0, 13.0, -5.0,    31.0, -5.0,    1.0e-18,
          18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 0.0,  -2.0, 2.0,  0.5 * 0.99});

  // Limit (2,1,2) because -(2,1,1)
  helper({0.0,  1.0,   2.0,  3.0,  4.0,  5.0, 6.0,  3.0, 3.0,
          9.0,  10.0,  11.0, 12.0, 13.0, 2.0, 15.0, 3.0, 4.1,
          18.0, -19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  2.5, 0.0},
         {0.0,  1.0,  2.0,  3.0,   4.0,  2.0, 6.0,  7.0, 3.0,
          9.0,  10.0, 11.0, 12.0,  13.0, 3.0, 15.0, 3.0, 4.1,
          18.0, 19.0, 3.0,  -21.0, 3.0,  4.1, 3.0,  4.1, 0.0},
         {0.0,  1.0,  2.0,   3.0,  4.0,  5.0, 6.0,   7.0, 3.0,
          9.0,  10.0, -11.0, 12.0, 13.0, 3.0, -15.0, 3.0, 4.1,
          18.0, 19.0, 3.0,   21.0, 3.0,  4.1, 3.0,   4.1, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  1.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  11.0,    -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -8.0,    1.0e-18,
          18.0, 49.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  8.0,     6.0,  7.0,     -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
          18.0, 19.0, -5.0, 43.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
          9.0,  10.0, 28.0, 12.0, 13.0, -3.75,   31.0, -5.0,    1.0e-18,
          18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,         6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0,        15.0, -2.0, 2.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 1.75 * 0.99, -2.0, 2.0,  0.5 * 0.99});

  // Zero (2,1,2) because -(2,1,1)
  helper({0.0,  1.0,   2.0,  3.0,  4.0,  5.0, 6.0,  3.0, 3.0,
          9.0,  10.0,  11.0, 12.0, 13.0, 2.0, 15.0, 3.0, 4.1,
          18.0, -19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  2.5, 0.0},
         {0.0,  1.0,  2.0,  3.0,   4.0,  2.0, 6.0,  7.0, 3.0,
          9.0,  10.0, 11.0, 12.0,  13.0, 3.0, 15.0, 3.0, 4.1,
          18.0, 19.0, 3.0,  -21.0, 3.0,  4.1, 3.0,  4.1, 0.0},
         {0.0,  1.0,  2.0,   3.0,  4.0,  5.0, 6.0,   7.0, 3.0,
          9.0,  10.0, -11.0, 12.0, 13.0, 3.0, -15.0, 3.0, 4.1,
          18.0, 19.0, 3.0,   21.0, 3.0,  4.1, 3.0,   4.1, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  1.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  11.0,    -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -8.0,    1.0e-18,
          18.0, 49.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  8.0,     6.0,  7.0,     -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
          18.0, 19.0, -5.0, 43.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
          9.0,  10.0, 28.0, 12.0, 13.0, -1.0,    31.0, -5.0,    1.0e-18,
          18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 0.0,  -2.0, 2.0,  0.5 * 0.99});
}

template <typename F>
void test_limiting_1_2_2_coefficient(const F& helper) noexcept {
  // Limit (1,2,2) because +(0,2,2)
  helper({0.0,  1.0,   2.0,  3.0,  4.0,  5.0, 6.0,   3.0, 3.0,
          9.0,  10.0,  11.0, 12.0, 13.0, 2.0, 15.0,  3.0, 4.1,
          18.0, -19.0, 3.0,  21.0, 3.0,  4.1, -0.64, 2.5, 0.0},
         {0.0,  1.0,  2.0,  3.0,   4.0,  2.0, 6.0,  7.0, 3.0,
          9.0,  10.0, 11.0, 12.0,  13.0, 3.0, 15.0, 3.0, 4.1,
          18.0, 19.0, 3.0,  -21.0, 3.0,  4.1, 3.0,  4.1, 0.0},
         {0.0,  1.0,  2.0,   3.0,  4.0,  5.0, 6.0,   7.0, 3.0,
          9.0,  10.0, -11.0, 12.0, 13.0, 3.0, -15.0, 3.0, 4.1,
          18.0, 19.0, 3.0,   21.0, 3.0,  4.1, 3.0,   4.1, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  1.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  11.0,    -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -8.0,    1.0e-18,
          18.0, 49.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  8.0,     6.0,  7.0,     -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
          18.0, 19.0, -5.0, 43.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
          9.0,  10.0, 28.0, 12.0, 13.0, -5.0,    31.0, -5.0,    1.0e-18,
          18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,         -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0,        2.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 1.36 * 0.99, 0.5 * 0.99});

  // Zero (1,2,2) because +(0,2,2)
  helper({0.0,  1.0,   2.0,  3.0,  4.0,  5.0, 6.0,  3.0, 3.0,
          9.0,  10.0,  11.0, 12.0, 13.0, 2.0, 15.0, 3.0, 4.1,
          18.0, -19.0, 3.0,  21.0, 3.0,  4.1, -8.0, 2.5, 0.0},
         {0.0,  1.0,  2.0,  3.0,   4.0,  2.0, 6.0,  7.0, 3.0,
          9.0,  10.0, 11.0, 12.0,  13.0, 3.0, 15.0, 3.0, 4.1,
          18.0, 19.0, 3.0,  -21.0, 3.0,  4.1, 3.0,  4.1, 0.0},
         {0.0,  1.0,  2.0,   3.0,  4.0,  5.0, 6.0,   7.0, 3.0,
          9.0,  10.0, -11.0, 12.0, 13.0, 3.0, -15.0, 3.0, 4.1,
          18.0, 19.0, 3.0,   21.0, 3.0,  4.1, 3.0,   4.1, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  1.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  11.0,    -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -8.0,    1.0e-18,
          18.0, 49.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  8.0,     6.0,  7.0,     -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
          18.0, 19.0, -5.0, 43.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
          9.0,  10.0, 28.0, 12.0, 13.0, -5.0,    31.0, -5.0,    1.0e-18,
          18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 0.0,  0.5 * 0.99});

  // Limit (1,2,2) because -(0,2,2)
  helper({0.0,  1.0,   2.0,  3.0,  4.0,  5.0, 6.0,  3.0, 3.0,
          9.0,  10.0,  11.0, 12.0, 13.0, 2.0, 15.0, 3.0, 4.1,
          18.0, -19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  2.5, 0.0},
         {0.0,  1.0,  2.0,  3.0,   4.0,  2.0, 6.0,  7.0, 3.0,
          9.0,  10.0, 11.0, 12.0,  13.0, 3.0, 15.0, 3.0, 4.1,
          18.0, 19.0, 3.0,  -21.0, 3.0,  4.1, 3.0,  4.1, 0.0},
         {0.0,  1.0,  2.0,   3.0,  4.0,  5.0, 6.0,   7.0, 3.0,
          9.0,  10.0, -11.0, 12.0, 13.0, 3.0, -15.0, 3.0, 4.1,
          18.0, 19.0, 3.0,   21.0, 3.0,  4.1, 3.0,   4.1, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  1.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,   11.0,    -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0,  -8.0,    1.0e-18,
          18.0, 49.0, -5.0, 21.0, -5.0, 1.0e-18, -3.13, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  8.0,     6.0,  7.0,     -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
          18.0, 19.0, -5.0, 43.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
          9.0,  10.0, 28.0, 12.0, 13.0, -5.0,    31.0, -5.0,    1.0e-18,
          18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,         -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0,        2.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 1.13 * 0.99, 0.5 * 0.99});

  // Zero (1,2,2) because -(0,2,2)
  helper({0.0,  1.0,   2.0,  3.0,  4.0,  5.0, 6.0,  3.0, 3.0,
          9.0,  10.0,  11.0, 12.0, 13.0, 2.0, 15.0, 3.0, 4.1,
          18.0, -19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  2.5, 0.0},
         {0.0,  1.0,  2.0,  3.0,   4.0,  2.0, 6.0,  7.0, 3.0,
          9.0,  10.0, 11.0, 12.0,  13.0, 3.0, 15.0, 3.0, 4.1,
          18.0, 19.0, 3.0,  -21.0, 3.0,  4.1, 3.0,  4.1, 0.0},
         {0.0,  1.0,  2.0,   3.0,  4.0,  5.0, 6.0,   7.0, 3.0,
          9.0,  10.0, -11.0, 12.0, 13.0, 3.0, -15.0, 3.0, 4.1,
          18.0, 19.0, 3.0,   21.0, 3.0,  4.1, 3.0,   4.1, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  1.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  11.0,    -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -8.0,    1.0e-18,
          18.0, 49.0, -5.0, 21.0, -5.0, 1.0e-18, -1.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  8.0,     6.0,  7.0,     -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
          18.0, 19.0, -5.0, 43.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
          9.0,  10.0, 28.0, 12.0, 13.0, -5.0,    31.0, -5.0,    1.0e-18,
          18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 0.0,  0.5 * 0.99});

  // Limit (1,2,2) because +(1,1,2)
  helper({0.0,  1.0,   2.0,  3.0,  4.0,  5.0, 6.0,  3.0, 3.0,
          9.0,  10.0,  11.0, 12.0, 13.0, 2.0, 15.0, 3.0, 4.1,
          18.0, -19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  2.5, 0.0},
         {0.0,  1.0,  2.0,  3.0,   4.0,   2.0, 6.0,  7.0, 3.0,
          9.0,  10.0, 11.0, 12.0,  13.0,  3.0, 15.0, 3.0, 4.1,
          18.0, 19.0, 3.0,  -21.0, -1.08, 4.1, 3.0,  4.1, 0.0},
         {0.0,  1.0,  2.0,   3.0,  4.0,  5.0, 6.0,   7.0, 3.0,
          9.0,  10.0, -11.0, 12.0, 13.0, 3.0, -15.0, 3.0, 4.1,
          18.0, 19.0, 3.0,   21.0, 3.0,  4.1, 3.0,   4.1, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  1.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  11.0,    -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -8.0,    1.0e-18,
          18.0, 49.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  8.0,     6.0,  7.0,     -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
          18.0, 19.0, -5.0, 43.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
          9.0,  10.0, 28.0, 12.0, 13.0, -5.0,    31.0, -5.0,    1.0e-18,
          18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,         -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0,        2.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 0.92 * 0.99, 0.5 * 0.99});

  // Skipping other (1,1,2) cases since they add limited value to the test

  // Limit (1,2,2) because +(1,2,1)
  helper({0.0,  1.0,   2.0,  3.0,  4.0,  5.0, 6.0,  3.0, 3.0,
          9.0,  10.0,  11.0, 12.0, 13.0, 2.0, 15.0, 3.0, 4.1,
          18.0, -19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  2.5, 0.0},
         {0.0,  1.0,  2.0,  3.0,   4.0,  2.0, 6.0,  7.0, 3.0,
          9.0,  10.0, 11.0, 12.0,  13.0, 3.0, 15.0, 3.0, 4.1,
          18.0, 19.0, 3.0,  -21.0, 3.0,  4.1, 3.0,  4.1, 0.0},
         {0.0,  1.0,  2.0,   3.0,  4.0,  5.0, 6.0,   7.0,   3.0,
          9.0,  10.0, -11.0, 12.0, 13.0, 3.0, -15.0, -1.03, 4.1,
          18.0, 19.0, 3.0,   21.0, 3.0,  4.1, 3.0,   4.1,   0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  1.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  11.0,    -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -8.0,    1.0e-18,
          18.0, 49.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  8.0,     6.0,  7.0,     -5.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
          18.0, 19.0, -5.0, 43.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
          9.0,  10.0, 28.0, 12.0, 13.0, -5.0,    31.0, -5.0,    1.0e-18,
          18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,         -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0,        2.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 0.97 * 0.99, 0.5 * 0.99});

  // Skipping other (1,2,1) cases since they add limited value to the test
}

template <typename F>
void test_limiting_0_1_2_coefficient_permutations(const F& helper) noexcept {
  // Limit (2,1,0) because -(2,0,0)
  helper({0.0,   10.0,  2.0,  3.0,   8.0,  5.0, 60.0,  3.0, 3.0,
          9.0,   100.0, 11.0, 120.0, 11.8, 2.0, -15.0, 3.0, 4.1,
          180.0, -19.0, 3.0,  -21.0, 3.0,  4.1, 3.0,   2.5, 0.0},
         {0.0,   1.0,   5.0,   30.0,  40.0,  2.0, 6.0,  7.0, 3.0,
          9.0,   100.0, -11.0, 120.0, -13.0, 3.0, 15.0, 3.0, 4.1,
          180.0, -19.0, 3.0,   -21.0, 3.0,   4.1, 3.0,  4.1, 0.0},
         {0.0,  1.0,   20.0, 3.0,   44.0,  4.0, 60.0,  -7.0,  3.0,
          90.0, 100.0, 10.0, 120.0, -13.0, 3.0, -15.0, -1.03, 4.1,
          18.0, 19.0,  3.0,  21.0,  3.0,   4.1, 3.0,   4.1,   0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  1.0},

         {0.0,    -10.0,  2.0,  3.0,    -4.0, 5.0,     -60.0, 11.0,    -5.0,
          9.0,    -100.0, 11.0, -120.0, 13.8, -5.0,    150.0, -8.0,    1.0e-18,
          -180.0, 49.0,   -5.0, 210.0,  -5.0, 1.0e-18, -5.0,  1.0e-18, 0.0},
         {0.0,    1.0,   1.5,  -30.0,  -40.0, 8.0,     6.0,  7.0,     -5.0,
          9.0,    -10.0, 11.5, -120.0, 130.0, -5.0,    15.0, -5.0,    1.0e-18,
          -180.0, 190.0, -5.0, 43.0,   -5.0,  1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,   1.0,    -20.0, 3.0,    -44.0, 8.0,     -60.0, 8.0,     -5.0,
          -90.0, -100.0, 28.0,  -120.0, 14.0,  -5.0,    31.0,  -5.0,    1.0e-18,
          18.0,  19.0,   -5.0,  21.0,   -5.0,  1.0e-18, -5.0,  1.0e-18, 0.0},

         {0.0,  1.0,         2.0,         3.0,  4.0,         0.5 * 0.99,
          6.0,  7.0,         -2.0,        9.0,  10.0,        11.0,
          12.0, 13.0,        -0.5 * 0.99, 15.0, -1.0 * 0.99, 2.0,
          18.0, 19.0,        -0.99,       21.0, -1.0 * 0.99, 2.0,
          -2.0, 0.97 * 0.99, 0.5 * 0.99});

  // Limit (1,2,0) because +(0,2,0)
  helper({0.0,   10.0,  2.0,  3.0,   80.0, 5.0, 8.0,   3.0, 3.0,
          9.0,   100.0, 11.0, 120.0, 11.8, 2.0, -15.0, 3.0, 4.1,
          180.0, -19.0, 3.0,  -21.0, 3.0,  4.1, 3.0,   2.5, 0.0},
         {0.0,   1.0,   50.0,  30.0,  40.0,  2.0, 6.0,  7.0, 3.0,
          9.0,   100.0, -11.0, 120.0, -13.0, 3.0, 15.0, 3.0, 4.1,
          180.0, -19.0, 3.0,   -21.0, 3.0,   4.1, 3.0,  4.1, 0.0},
         {0.0,  1.0,   20.0, 3.0,   44.0,  4.0, 60.0,  -7.0,  3.0,
          90.0, 100.0, 10.0, 120.0, -13.0, 3.0, -15.0, -1.03, 4.1,
          18.0, 19.0,  3.0,  21.0,  3.0,   4.1, 3.0,   4.1,   0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  1.0},

         {0.0,    -10.0,  2.0,  3.0,    -4.0, 5.0,     -60.0, 11.0,    -5.0,
          9.0,    -100.0, 11.0, -120.0, 13.8, -5.0,    150.0, -8.0,    1.0e-18,
          -180.0, 49.0,   -5.0, 210.0,  -5.0, 1.0e-18, -5.0,  1.0e-18, 0.0},
         {0.0,    1.0,   -8.0, -30.0,  -40.0, 8.0,     6.0,  7.0,     -5.0,
          9.0,    -10.0, 11.5, -120.0, 130.0, -5.0,    15.0, -5.0,    1.0e-18,
          -180.0, 190.0, -8.0, 43.0,   -5.0,  1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,   1.0,    -20.0, 3.0,    -44.0, 8.0,     -60.0, 8.0,     -5.0,
          -90.0, -100.0, 28.0,  -120.0, 14.0,  -5.0,    31.0,  -5.0,    1.0e-18,
          18.0,  19.0,   -5.0,  21.0,   -5.0,  1.0e-18, -5.0,  1.0e-18, 0.0},

         {0.0,  1.0,         2.0,         3.0,  4.0,         5.0,
          6.0,  2.0 * 0.99,  -2.0,        9.0,  10.0,        11.0,
          12.0, 13.0,        -0.5 * 0.99, 15.0, -1.0 * 0.99, 2.0,
          18.0, 19.0,        -0.99,       21.0, -1.0 * 0.99, 2.0,
          -2.0, 0.97 * 0.99, 0.5 * 0.99});

  // Limit (2,0,1) because +(2,0,0)
  helper({0.0,   10.0,  2.0,  3.0,   80.0, 5.0, 60.0,  3.0, 3.0,
          9.0,   100.0, 11.0, 120.0, 11.8, 2.0, -15.0, 3.0, 4.1,
          180.0, -19.0, 3.0,  -21.0, 3.0,  4.1, 3.0,   2.5, 0.0},
         {0.0,   1.0,   50.0,  30.0,  40.0,  2.0, 6.0,  7.0, 3.0,
          9.0,   100.0, -11.0, 120.0, -13.0, 3.0, 15.0, 3.0, 4.1,
          180.0, -19.0, 3.0,   -21.0, 3.0,   4.1, 3.0,  4.1, 0.0},
         {0.0,  1.0,   3.0,  3.0,   44.0,  4.0, 60.0,  -7.0,  3.0,
          90.0, 100.0, 10.0, 120.0, -13.0, 3.0, -15.0, -1.03, 4.1,
          18.0, 19.0,  3.0,  21.0,  3.0,   4.1, 3.0,   4.1,   0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  1.0},

         {0.0,    -10.0,  2.0,  3.0,    -4.0, 5.0,     -60.0, 11.0,    -5.0,
          9.0,    -100.0, 11.0, -120.0, 13.8, -5.0,    150.0, -8.0,    1.0e-18,
          -180.0, 49.0,   -5.0, 210.0,  -5.0, 1.0e-18, -5.0,  1.0e-18, 0.0},
         {0.0,    1.0,   -8.0, -30.0,  -40.0, 8.0,     6.0,  7.0,     -5.0,
          9.0,    -10.0, 11.5, -120.0, 130.0, -5.0,    15.0, -5.0,    1.0e-18,
          -180.0, 190.0, -8.0, 43.0,   -5.0,  1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,   1.0,    -20.0, 3.0,    -44.0, 8.0,     -60.0, 8.0,     -5.0,
          -90.0, -100.0, 28.0,  -120.0, 14.0,  -5.0,    31.0,  -5.0,    1.0e-18,
          18.0,  19.0,   -5.0,  21.0,   -5.0,  1.0e-18, -5.0,  1.0e-18, 0.0},

         {0.0,  1.0,         2.0,         3.0,  4.0,         5.0,
          6.0,  7.0,         -2.0,        9.0,  10.0,        1.0 * 0.99,
          12.0, 13.0,        -0.5 * 0.99, 15.0, -1.0 * 0.99, 2.0,
          18.0, 19.0,        -0.99,       21.0, -1.0 * 0.99, 2.0,
          -2.0, 0.97 * 0.99, 0.5 * 0.99});

  // Limit (1,0,2) because -(0,0,2)
  helper({0.0,   10.0,  2.0,  3.0,   80.0, 5.0, 60.0,  3.0, 3.0,
          9.0,   100.0, 11.0, 120.0, 11.8, 2.0, -15.0, 3.0, 4.1,
          180.0, -19.0, 3.0,  -21.0, 3.0,  4.1, 3.0,   2.5, 0.0},
         {0.0,   1.0,   50.0,  30.0,  40.0,  2.0, 6.0,  7.0, 3.0,
          9.0,   100.0, -11.0, 120.0, -13.0, 3.0, 15.0, 3.0, 4.1,
          180.0, -19.0, 3.0,   -21.0, 3.0,   4.1, 3.0,  4.1, 0.0},
         {0.0,  1.0,   20.0, 3.0,   44.0,  4.0, 60.0,  -7.0,  3.0,
          90.0, 100.0, 10.0, 120.0, -13.0, 3.0, -15.0, -1.03, 4.1,
          18.0, 19.0,  3.0,  21.0,  3.0,   4.1, 3.0,   4.1,   0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  1.0},

         {0.0,  -10.0,  2.0,  3.0,    -4.0, 5.0,     -60.0, 11.0,    -5.0,
          9.0,  -100.0, 11.0, -120.0, 13.8, -5.0,    150.0, -8.0,    1.0e-18,
          14.0, 49.0,   -5.0, 210.0,  -5.0, 1.0e-18, -5.0,  1.0e-18, 0.0},
         {0.0,    1.0,   -8.0, -30.0,  -40.0, 8.0,     6.0,  7.0,     -5.0,
          9.0,    -10.0, 11.5, -120.0, 130.0, -5.0,    15.0, -5.0,    1.0e-18,
          -180.0, 190.0, -8.0, 43.0,   -5.0,  1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,   1.0,    -20.0, 3.0,    -44.0, 8.0,     -60.0, 8.0,     -5.0,
          -90.0, -100.0, 28.0,  -120.0, 14.0,  -5.0,    31.0,  -5.0,    1.0e-18,
          18.0,  19.0,   -5.0,  21.0,   -5.0,  1.0e-18, -5.0,  1.0e-18, 0.0},

         {0.0,  1.0,         2.0,         3.0,  4.0,         5.0,
          6.0,  7.0,         -2.0,        9.0,  10.0,        11.0,
          12.0, 13.0,        -0.5 * 0.99, 15.0, -1.0 * 0.99, 2.0,
          18.0, 4.0 * 0.99,  -0.99,       21.0, -1.0 * 0.99, 2.0,
          -2.0, 0.97 * 0.99, 0.5 * 0.99});

  // Limit (0,2,1) because +(0,2,0)
  helper({0.0,   10.0,  2.0,  3.0,   80.0, 5.0, 60.0,  3.0, 3.0,
          9.0,   100.0, 11.0, 120.0, 11.8, 2.0, -15.0, 3.0, 4.1,
          180.0, -19.0, 3.0,  -21.0, 3.0,  4.1, 3.0,   2.5, 0.0},
         {0.0,   1.0,   50.0,  30.0,  40.0,  2.0, 6.0,  7.0, 3.0,
          9.0,   100.0, -11.0, 120.0, -13.0, 3.0, 15.0, 3.0, 4.1,
          180.0, -19.0, 3.0,   -21.0, 3.0,   4.1, 3.0,  4.1, 0.0},
         {0.0,  1.0,   20.0, 3.0,   44.0,  4.0, 8.0,   -7.0,  3.0,
          90.0, 100.0, 10.0, 120.0, -13.0, 3.0, -15.0, -1.03, 4.1,
          18.0, 19.0,  3.0,  21.0,  3.0,   4.1, 3.0,   4.1,   0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  1.0},

         {0.0,    -10.0,  2.0,  3.0,    -4.0, 5.0,     -60.0, 11.0,    -5.0,
          9.0,    -100.0, 11.0, -120.0, 13.8, -5.0,    150.0, -8.0,    1.0e-18,
          -180.0, 49.0,   -5.0, 210.0,  -5.0, 1.0e-18, -5.0,  1.0e-18, 0.0},
         {0.0,    1.0,   -8.0, -30.0,  -40.0, 8.0,     6.0,  7.0,     -5.0,
          9.0,    -10.0, 11.5, -120.0, 130.0, -5.0,    15.0, -5.0,    1.0e-18,
          -180.0, 190.0, -8.0, 43.0,   -5.0,  1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,   1.0,    -20.0, 3.0,    -44.0, 8.0,     -60.0, 8.0,     -5.0,
          -90.0, -100.0, 28.0,  -120.0, 14.0,  -5.0,    31.0,  -5.0,    1.0e-18,
          18.0,  19.0,   -5.0,  21.0,   -5.0,  1.0e-18, -5.0,  1.0e-18, 0.0},

         {0.0,  1.0,         2.0,         3.0,        4.0,         5.0,
          6.0,  7.0,         -2.0,        9.0,        10.0,        11.0,
          12.0, 13.0,        -0.5 * 0.99, 2.0 * 0.99, -1.0 * 0.99, 2.0,
          18.0, 19.0,        -0.99,       21.0,       -1.0 * 0.99, 2.0,
          -2.0, 0.97 * 0.99, 0.5 * 0.99});

  // Limit (0,1, 2) because +(0,0,2)
  helper({0.0,   10.0,  2.0,  3.0,   80.0, 5.0, 60.0,  3.0, 3.0,
          9.0,   100.0, 11.0, 120.0, 11.8, 2.0, -15.0, 3.0, 4.1,
          180.0, -19.0, 3.0,  -21.0, 3.0,  4.1, 3.0,   2.5, 0.0},
         {0.0,  1.0,   50.0,  30.0,  40.0,  2.0, 6.0,  7.0, 3.0,
          9.0,  100.0, -11.0, 120.0, -13.0, 3.0, 15.0, 3.0, 4.1,
          21.0, -19.0, 3.0,   -21.0, 3.0,   4.1, 3.0,  4.1, 0.0},
         {0.0,  1.0,   20.0, 3.0,   44.0,  4.0, 60.0,  -7.0,  3.0,
          90.0, 100.0, 10.0, 120.0, -13.0, 3.0, -15.0, -1.03, 4.1,
          18.0, 19.0,  3.0,  21.0,  3.0,   4.1, 3.0,   4.1,   0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  1.0},

         {0.0,    -10.0,  2.0,  3.0,    -4.0, 5.0,     -60.0, 11.0,    -5.0,
          9.0,    -100.0, 11.0, -120.0, 13.8, -5.0,    150.0, -8.0,    1.0e-18,
          -180.0, 49.0,   -5.0, 210.0,  -5.0, 1.0e-18, -5.0,  1.0e-18, 0.0},
         {0.0,    1.0,   -8.0, -30.0,  -40.0, 8.0,     6.0,  7.0,     -5.0,
          9.0,    -10.0, 11.5, -120.0, 130.0, -5.0,    15.0, -5.0,    1.0e-18,
          -180.0, 190.0, -8.0, 43.0,   -5.0,  1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,   1.0,    -20.0, 3.0,    -44.0, 8.0,     -60.0, 8.0,     -5.0,
          -90.0, -100.0, 28.0,  -120.0, 14.0,  -5.0,    31.0,  -5.0,    1.0e-18,
          18.0,  19.0,   -5.0,  21.0,   -5.0,  1.0e-18, -5.0,  1.0e-18, 0.0},

         {0.0,  1.0,         2.0,         3.0,        4.0,         5.0,
          6.0,  7.0,         -2.0,        9.0,        10.0,        11.0,
          12.0, 13.0,        -0.5 * 0.99, 15.0,       -1.0 * 0.99, 2.0,
          18.0, 19.0,        -0.99,       3.0 * 0.99, -1.0 * 0.99, 2.0,
          -2.0, 0.97 * 0.99, 0.5 * 0.99});

  // Don't limit (0,1,2) permutations to verify we stop correctly
  helper({0.0,   10.0,  2.0,  3.0,   80.0, 5.0, 60.0,  3.0, 3.0,
          9.0,   100.0, 11.0, 120.0, 11.8, 2.0, -15.0, 3.0, 4.1,
          180.0, -19.0, 3.0,  -21.0, 3.0,  4.1, 3.0,   2.5, 0.0},
         {0.0,   1.0,   50.0,  30.0,  40.0,  2.0, 6.0,  7.0, 3.0,
          9.0,   100.0, -11.0, 120.0, -13.0, 3.0, 15.0, 3.0, 4.1,
          180.0, -19.0, 3.0,   -21.0, 3.0,   4.1, 3.0,  4.1, 0.0},
         {0.0,  1.0,   20.0, 3.0,   44.0,  4.0, 60.0,  -7.0,  3.0,
          90.0, 100.0, 10.0, 120.0, -13.0, 3.0, -15.0, -1.03, 4.1,
          18.0, 19.0,  3.0,  21.0,  3.0,   4.1, 3.0,   4.1,   0.0},

         {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
          9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
          18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  1.0},

         {0.0,    -10.0,  2.0,  3.0,    -4.0, 5.0,     -60.0, 11.0,    -5.0,
          9.0,    -100.0, 11.0, -120.0, 13.8, -5.0,    150.0, -8.0,    1.0e-18,
          -180.0, 49.0,   -5.0, 210.0,  -5.0, 1.0e-18, -5.0,  1.0e-18, 0.0},
         {0.0,    1.0,   -8.0, -30.0,  -40.0, 8.0,     6.0,  7.0,     -5.0,
          9.0,    -10.0, 11.5, -120.0, 130.0, -5.0,    15.0, -5.0,    1.0e-18,
          -180.0, 190.0, -8.0, 43.0,   -5.0,  1.0e-18, -5.0, 1.0e-18, 0.0},
         {0.0,   1.0,    -20.0, 3.0,    -44.0, 8.0,     -60.0, 8.0,     -5.0,
          -90.0, -100.0, 28.0,  -120.0, 14.0,  -5.0,    31.0,  -5.0,    1.0e-18,
          18.0,  19.0,   -5.0,  21.0,   -5.0,  1.0e-18, -5.0,  1.0e-18, 0.0},

         {0.0,  1.0,         2.0,         3.0,  4.0,         5.0,
          6.0,  7.0,         -2.0,        9.0,  10.0,        11.0,
          12.0, 13.0,        -0.5 * 0.99, 15.0, -1.0 * 0.99, 2.0,
          18.0, 19.0,        -0.99,       21.0, -1.0 * 0.99, 2.0,
          -2.0, 0.97 * 0.99, 0.5 * 0.99});
}

void test_limiting_different_values_different_tensors() noexcept {
  INFO("Testing different values for each tensor component");
  constexpr size_t dim = 3;
  const size_t order = 2;  // Use only 3 coefficients because more is tedious...
  const Mesh<dim> mesh(order + 1, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto);
  const size_t num_pts = mesh.number_of_grid_points();
  const auto x = logical_coordinates(mesh);
  using Limiter = Krivodonova<dim, tmpl::list<ScalarTag<0>, VectorTag<dim, 0>>>;
  // Use non-unity (because that's the default) but close alpha values to make
  // the math easier but still test thoroughly.
  Limiter krivodonova{
      make_array<Spectral::maximum_number_of_points<Spectral::Basis::Legendre>>(
          0.99)};

  NeighborData<dim, typename Limiter::PackagedData> neighbor_data{};

  const Element<dim> element(ElementId<dim>{0}, {});
  // We don't care about the ElementId for these tests, just the direction.
  Limiter::PackagedData& package_data_up_xi = neighbor_data[std::make_pair(
      Direction<dim>::upper_xi(), ElementId<dim>{0})];
  Limiter::PackagedData& package_data_lo_xi = neighbor_data[std::make_pair(
      Direction<dim>::lower_xi(), ElementId<dim>{0})];
  Limiter::PackagedData& package_data_up_eta = neighbor_data[std::make_pair(
      Direction<dim>::upper_eta(), ElementId<dim>{0})];
  Limiter::PackagedData& package_data_lo_eta = neighbor_data[std::make_pair(
      Direction<dim>::lower_eta(), ElementId<dim>{0})];
  Limiter::PackagedData& package_data_up_zeta = neighbor_data[std::make_pair(
      Direction<dim>::upper_zeta(), ElementId<dim>{0})];
  Limiter::PackagedData& package_data_lo_zeta = neighbor_data[std::make_pair(
      Direction<dim>::lower_zeta(), ElementId<dim>{0})];

  package_data_up_xi.modal_volume_data.initialize(num_pts);
  package_data_up_xi.mesh = mesh;
  package_data_lo_xi.modal_volume_data.initialize(num_pts);
  package_data_lo_xi.mesh = mesh;
  package_data_up_eta.modal_volume_data.initialize(num_pts);
  package_data_up_eta.mesh = mesh;
  package_data_lo_eta.modal_volume_data.initialize(num_pts);
  package_data_lo_eta.mesh = mesh;
  package_data_up_zeta.modal_volume_data.initialize(num_pts);
  package_data_up_zeta.mesh = mesh;
  package_data_lo_zeta.modal_volume_data.initialize(num_pts);
  package_data_lo_zeta.mesh = mesh;

  Scalar<DataVector> nodal_scalar_data_to_limit(num_pts, 0.0);
  tnsr::I<DataVector, dim> nodal_vector_data_to_limit(num_pts, 0.0);
  Scalar<DataVector> nodal_scalar_expected(num_pts, 0.0);
  tnsr::I<DataVector, dim> nodal_vector_expected(num_pts, 0.0);

  // Limit (2,2,1) because +(1,2,1)
  to_nodal_coefficients(&get(nodal_scalar_data_to_limit),
                        {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
                         9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
                         18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  1.0},
                        mesh);
  get(get<::Tags::Modal<ScalarTag<0>>>(package_data_up_xi.modal_volume_data)) =
      ModalVector{0.0,  1.0,   2.0,  3.0,  4.0,  5.0, 6.0,  3.0,  3.0,
                  9.0,  10.0,  11.0, 12.0, 13.0, 2.0, 15.0, -1.1, 4.1,
                  18.0, -19.0, 3.0,  21.0, 3.0,  4.1, 3.0,  2.5,  0.0};
  get(get<::Tags::Modal<ScalarTag<0>>>(package_data_lo_xi.modal_volume_data)) =
      ModalVector{0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  11.0,    -5.0,
                  9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -8.0,    1.0e-18,
                  18.0, 49.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0};
  get(get<::Tags::Modal<ScalarTag<0>>>(package_data_up_eta.modal_volume_data)) =
      ModalVector{0.0,  1.0,  2.0,  3.0,   4.0,  2.0, 6.0,  7.0, 3.0,
                  9.0,  10.0, 11.0, 12.0,  13.0, 3.0, 15.0, 3.0, 4.1,
                  18.0, 19.0, 3.0,  -21.0, 3.0,  4.1, 3.0,  4.1, 0.0};
  get(get<::Tags::Modal<ScalarTag<0>>>(package_data_lo_eta.modal_volume_data)) =
      ModalVector{0.0,  1.0,  2.0,  3.0,  4.0,  8.0,     6.0,  7.0,     -5.0,
                  9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
                  18.0, 19.0, -5.0, 43.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0};
  get(get<::Tags::Modal<ScalarTag<0>>>(
      package_data_up_zeta.modal_volume_data)) =
      ModalVector{0.0,  1.0,  2.0,   3.0,  4.0,  5.0, 6.0,   7.0, 3.0,
                  9.0,  10.0, -11.0, 12.0, 13.0, 3.0, -15.0, 3.0, 4.1,
                  18.0, 19.0, 3.0,   21.0, 3.0,  4.1, 3.0,   4.1, 0.0};
  get(get<::Tags::Modal<ScalarTag<0>>>(
      package_data_lo_zeta.modal_volume_data)) =
      ModalVector{0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
                  9.0,  10.0, 28.0, 12.0, 13.0, -5.0,    31.0, -5.0,    1.0e-18,
                  18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0};
  to_nodal_coefficients(
      &get(nodal_scalar_expected),
      {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
       9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 0.9 * 0.99,
       18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  0.5 * 0.99},
      mesh);

  // Limit (2,1,2) because +(1,1,2)
  to_nodal_coefficients(&nodal_vector_data_to_limit.get(0),
                        {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
                         9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
                         18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  1.0},
                        mesh);
  get<::Tags::Modal<VectorTag<dim, 0>>>(package_data_up_xi.modal_volume_data)
      .get(0) =
      ModalVector{0.0,  1.0,   2.0,  3.0,  4.0,   5.0, 6.0,  3.0, 3.0,
                  9.0,  10.0,  11.0, 12.0, 13.0,  2.0, 15.0, 3.0, 4.1,
                  18.0, -19.0, 3.0,  21.0, -1.15, 4.1, 3.0,  2.5, 0.0};
  get<::Tags::Modal<VectorTag<dim, 0>>>(package_data_lo_xi.modal_volume_data)
      .get(0) =
      ModalVector{0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  11.0,    -5.0,
                  9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -8.0,    1.0e-18,
                  18.0, 49.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0};
  get<::Tags::Modal<VectorTag<dim, 0>>>(package_data_up_eta.modal_volume_data)
      .get(0) = ModalVector{0.0,  1.0,  2.0,  3.0,   4.0,  2.0, 6.0,  7.0, 3.0,
                            9.0,  10.0, 11.0, 12.0,  13.0, 3.0, 15.0, 3.0, 4.1,
                            18.0, 19.0, 3.0,  -21.0, 3.0,  4.1, 3.0,  4.1, 0.0};
  get<::Tags::Modal<VectorTag<dim, 0>>>(package_data_lo_eta.modal_volume_data)
      .get(0) =
      ModalVector{0.0,  1.0,  2.0,  3.0,  4.0,  8.0,     6.0,  7.0,     -5.0,
                  9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
                  18.0, 19.0, -5.0, 43.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0};
  get<::Tags::Modal<VectorTag<dim, 0>>>(package_data_up_zeta.modal_volume_data)
      .get(0) =
      ModalVector{0.0,  1.0,  2.0,   3.0,  4.0,  5.0, 6.0,   7.0, 3.0,
                  9.0,  10.0, -11.0, 12.0, 13.0, 3.0, -15.0, 3.0, 4.1,
                  18.0, 19.0, 3.0,   21.0, 3.0,  4.1, 3.0,   4.1, 0.0};
  get<::Tags::Modal<VectorTag<dim, 0>>>(package_data_lo_zeta.modal_volume_data)
      .get(0) =
      ModalVector{0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
                  9.0,  10.0, 28.0, 12.0, 13.0, -5.0,    31.0, -5.0,    1.0e-18,
                  18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0};
  to_nodal_coefficients(
      &nodal_vector_expected.get(0),
      {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,         6.0,  7.0,  -2.0,
       9.0,  10.0, 11.0, 12.0, 13.0, -2.0,        15.0, -2.0, 2.0,
       18.0, 19.0, -2.0, 21.0, -2.0, 0.85 * 0.99, -2.0, 2.0,  0.5 * 0.99},
      mesh);

  // Limit (1,2,2) because +(0,2,2)
  to_nodal_coefficients(&nodal_vector_data_to_limit.get(1),
                        {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
                         9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
                         18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  1.0},
                        mesh);
  get<::Tags::Modal<VectorTag<dim, 0>>>(package_data_up_xi.modal_volume_data)
      .get(1) =
      ModalVector{0.0,  1.0,   2.0,  3.0,  4.0,  5.0, 6.0,   3.0, 3.0,
                  9.0,  10.0,  11.0, 12.0, 13.0, 2.0, 15.0,  3.0, 4.1,
                  18.0, -19.0, 3.0,  21.0, 3.0,  4.1, -0.64, 2.5, 0.0};
  get<::Tags::Modal<VectorTag<dim, 0>>>(package_data_lo_xi.modal_volume_data)
      .get(1) =
      ModalVector{0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  11.0,    -5.0,
                  9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -8.0,    1.0e-18,
                  18.0, 49.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0};
  get<::Tags::Modal<VectorTag<dim, 0>>>(package_data_up_eta.modal_volume_data)
      .get(1) = ModalVector{0.0,  1.0,  2.0,  3.0,   4.0,  2.0, 6.0,  7.0, 3.0,
                            9.0,  10.0, 11.0, 12.0,  13.0, 3.0, 15.0, 3.0, 4.1,
                            18.0, 19.0, 3.0,  -21.0, 3.0,  4.1, 3.0,  4.1, 0.0};
  get<::Tags::Modal<VectorTag<dim, 0>>>(package_data_lo_eta.modal_volume_data)
      .get(1) =
      ModalVector{0.0,  1.0,  2.0,  3.0,  4.0,  8.0,     6.0,  7.0,     -5.0,
                  9.0,  10.0, 11.0, 12.0, 13.0, -5.0,    15.0, -5.0,    1.0e-18,
                  18.0, 19.0, -5.0, 43.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0};
  get<::Tags::Modal<VectorTag<dim, 0>>>(package_data_up_zeta.modal_volume_data)
      .get(1) =
      ModalVector{0.0,  1.0,  2.0,   3.0,  4.0,  5.0, 6.0,   7.0, 3.0,
                  9.0,  10.0, -11.0, 12.0, 13.0, 3.0, -15.0, 3.0, 4.1,
                  18.0, 19.0, 3.0,   21.0, 3.0,  4.1, 3.0,   4.1, 0.0};
  get<::Tags::Modal<VectorTag<dim, 0>>>(package_data_lo_zeta.modal_volume_data)
      .get(1) =
      ModalVector{0.0,  1.0,  2.0,  3.0,  4.0,  5.0,     6.0,  7.0,     -5.0,
                  9.0,  10.0, 28.0, 12.0, 13.0, -5.0,    31.0, -5.0,    1.0e-18,
                  18.0, 19.0, -5.0, 21.0, -5.0, 1.0e-18, -5.0, 1.0e-18, 0.0};
  to_nodal_coefficients(
      &nodal_vector_expected.get(1),
      {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,         -2.0,
       9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0,        2.0,
       18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 1.36 * 0.99, 0.5 * 0.99},
      mesh);

  // Limit (2,1,0) because -(2,0,0)
  to_nodal_coefficients(&nodal_vector_data_to_limit.get(2),
                        {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  -2.0,
                         9.0,  10.0, 11.0, 12.0, 13.0, -2.0, 15.0, -2.0, 2.0,
                         18.0, 19.0, -2.0, 21.0, -2.0, 2.0,  -2.0, 2.0,  1.0},
                        mesh);
  get<::Tags::Modal<VectorTag<dim, 0>>>(package_data_up_xi.modal_volume_data)
      .get(2) =
      ModalVector{0.0,   10.0,  2.0,  3.0,   8.0,  5.0, 60.0,  3.0, 3.0,
                  9.0,   100.0, 11.0, 120.0, 11.8, 2.0, -15.0, 3.0, 4.1,
                  180.0, -19.0, 3.0,  -21.0, 3.0,  4.1, 3.0,   2.5, 0.0};
  get<::Tags::Modal<VectorTag<dim, 0>>>(package_data_lo_xi.modal_volume_data)
      .get(2) = ModalVector{
      0.0,    -10.0,  2.0,  3.0,    -4.0, 5.0,     -60.0, 11.0,    -5.0,
      9.0,    -100.0, 11.0, -120.0, 13.8, -5.0,    150.0, -8.0,    1.0e-18,
      -180.0, 49.0,   -5.0, 210.0,  -5.0, 1.0e-18, -5.0,  1.0e-18, 0.0};
  get<::Tags::Modal<VectorTag<dim, 0>>>(package_data_up_eta.modal_volume_data)
      .get(2) =
      ModalVector{0.0,   1.0,   5.0,   30.0,  40.0,  2.0, 6.0,  7.0, 3.0,
                  9.0,   100.0, -11.0, 120.0, -13.0, 3.0, 15.0, 3.0, 4.1,
                  180.0, -19.0, 3.0,   -21.0, 3.0,   4.1, 3.0,  4.1, 0.0};
  get<::Tags::Modal<VectorTag<dim, 0>>>(package_data_lo_eta.modal_volume_data)
      .get(2) = ModalVector{
      0.0,    1.0,   1.5,  -30.0,  -40.0, 8.0,     6.0,  7.0,     -5.0,
      9.0,    -10.0, 11.5, -120.0, 130.0, -5.0,    15.0, -5.0,    1.0e-18,
      -180.0, 190.0, -5.0, 43.0,   -5.0,  1.0e-18, -5.0, 1.0e-18, 0.0};
  get<::Tags::Modal<VectorTag<dim, 0>>>(package_data_up_zeta.modal_volume_data)
      .get(2) =
      ModalVector{0.0,  1.0,   20.0, 3.0,   44.0,  4.0, 60.0,  -7.0,  3.0,
                  90.0, 100.0, 10.0, 120.0, -13.0, 3.0, -15.0, -1.03, 4.1,
                  18.0, 19.0,  3.0,  21.0,  3.0,   4.1, 3.0,   4.1,   0.0};
  get<::Tags::Modal<VectorTag<dim, 0>>>(package_data_lo_zeta.modal_volume_data)
      .get(2) = ModalVector{
      0.0,   1.0,    -20.0, 3.0,    -44.0, 8.0,     -60.0, 8.0,     -5.0,
      -90.0, -100.0, 28.0,  -120.0, 14.0,  -5.0,    31.0,  -5.0,    1.0e-18,
      18.0,  19.0,   -5.0,  21.0,   -5.0,  1.0e-18, -5.0,  1.0e-18, 0.0};
  to_nodal_coefficients(
      &nodal_vector_expected.get(2),
      {0.0,  1.0,         2.0,         3.0,  4.0,         0.5 * 0.99,
       6.0,  7.0,         -2.0,        9.0,  10.0,        11.0,
       12.0, 13.0,        -0.5 * 0.99, 15.0, -1.0 * 0.99, 2.0,
       18.0, 19.0,        -0.99,       21.0, -1.0 * 0.99, 2.0,
       -2.0, 0.97 * 0.99, 0.5 * 0.99},
      mesh);

  krivodonova(&nodal_scalar_data_to_limit, &nodal_vector_data_to_limit, element,
              mesh, neighbor_data);
  CHECK_ITERABLE_APPROX(get(nodal_scalar_data_to_limit),
                        get(nodal_scalar_expected));
  for (size_t i = 0; i < dim; ++i) {
    CAPTURE(i);
    CHECK_ITERABLE_APPROX(nodal_vector_data_to_limit.get(i),
                          nodal_vector_expected.get(i));
  }
}

void run() noexcept {
  INFO("Testing 3d limiter");
  test_package_data();

  INFO("Testing applying limiter to coefficients");
  constexpr size_t dim = 3;
  const size_t order = 2;  // Use only 3 coefficients because more is tedious...
  const Mesh<dim> mesh(order + 1, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto);
  const size_t num_pts = mesh.number_of_grid_points();
  using Limiter = Krivodonova<dim, tmpl::list<ScalarTag<0>, VectorTag<dim, 0>>>;
  // Use non-unity (because that's the default) but close alpha values to make
  // the math easier but still test thoroughly.
  Limiter krivodonova{
      make_array<Spectral::maximum_number_of_points<Spectral::Basis::Legendre>>(
          0.99)};

  NeighborData<dim, typename Limiter::PackagedData> neighbor_data{};

  const Element<dim> element(ElementId<dim>{0}, {});
  // We don't care about the ElementId for these tests, just the direction.
  Limiter::PackagedData& package_data_up_xi = neighbor_data[std::make_pair(
      Direction<dim>::upper_xi(), ElementId<dim>{0})];
  Limiter::PackagedData& package_data_lo_xi = neighbor_data[std::make_pair(
      Direction<dim>::lower_xi(), ElementId<dim>{0})];
  Limiter::PackagedData& package_data_up_eta = neighbor_data[std::make_pair(
      Direction<dim>::upper_eta(), ElementId<dim>{0})];
  Limiter::PackagedData& package_data_lo_eta = neighbor_data[std::make_pair(
      Direction<dim>::lower_eta(), ElementId<dim>{0})];
  Limiter::PackagedData& package_data_up_zeta = neighbor_data[std::make_pair(
      Direction<dim>::upper_zeta(), ElementId<dim>{0})];
  Limiter::PackagedData& package_data_lo_zeta = neighbor_data[std::make_pair(
      Direction<dim>::lower_zeta(), ElementId<dim>{0})];

  package_data_up_xi.modal_volume_data.initialize(num_pts);
  package_data_up_xi.mesh = mesh;
  package_data_lo_xi.modal_volume_data.initialize(num_pts);
  package_data_lo_xi.mesh = mesh;
  package_data_up_eta.modal_volume_data.initialize(num_pts);
  package_data_up_eta.mesh = mesh;
  package_data_lo_eta.modal_volume_data.initialize(num_pts);
  package_data_lo_eta.mesh = mesh;
  package_data_up_zeta.modal_volume_data.initialize(num_pts);
  package_data_up_zeta.mesh = mesh;
  package_data_lo_zeta.modal_volume_data.initialize(num_pts);
  package_data_lo_zeta.mesh = mesh;

  Scalar<DataVector> nodal_scalar_data_to_limit(num_pts, 0.0);
  tnsr::I<DataVector, dim> nodal_vector_data_to_limit(num_pts, 0.0);
  DataVector expected(num_pts);
  const auto helper =
      [
        &element, &expected, &krivodonova, &mesh, &neighbor_data,
        &package_data_lo_xi, &package_data_up_xi, &package_data_lo_eta,
        &package_data_up_eta, &package_data_lo_zeta, &package_data_up_zeta,
        &nodal_scalar_data_to_limit, &nodal_vector_data_to_limit
      ](const ModalVector& up_xi_coeffs, const ModalVector& up_eta_coeffs,
        const ModalVector& up_zeta_coeffs, const ModalVector& initial_coeffs,
        const ModalVector& lo_xi_coeffs, const ModalVector& lo_eta_coeffs,
        const ModalVector& lo_zeta_coeffs,
        const ModalVector& expected_coeffs) noexcept {
    to_nodal_coefficients(&get(nodal_scalar_data_to_limit), initial_coeffs,
                          mesh);
    get(get<::Tags::Modal<ScalarTag<0>>>(
        package_data_up_xi.modal_volume_data)) = up_xi_coeffs;
    get(get<::Tags::Modal<ScalarTag<0>>>(
        package_data_lo_xi.modal_volume_data)) = lo_xi_coeffs;
    get(get<::Tags::Modal<ScalarTag<0>>>(
        package_data_up_eta.modal_volume_data)) = up_eta_coeffs;
    get(get<::Tags::Modal<ScalarTag<0>>>(
        package_data_lo_eta.modal_volume_data)) = lo_eta_coeffs;
    get(get<::Tags::Modal<ScalarTag<0>>>(
        package_data_up_zeta.modal_volume_data)) = up_zeta_coeffs;
    get(get<::Tags::Modal<ScalarTag<0>>>(
        package_data_lo_zeta.modal_volume_data)) = lo_zeta_coeffs;

    for (size_t i = 0; i < dim; ++i) {
      to_nodal_coefficients(&nodal_vector_data_to_limit.get(i), initial_coeffs,
                            mesh);
      get<::Tags::Modal<VectorTag<dim, 0>>>(
          package_data_up_xi.modal_volume_data)
          .get(i) = up_xi_coeffs;
      get<::Tags::Modal<VectorTag<dim, 0>>>(
          package_data_lo_xi.modal_volume_data)
          .get(i) = lo_xi_coeffs;
      get<::Tags::Modal<VectorTag<dim, 0>>>(
          package_data_up_eta.modal_volume_data)
          .get(i) = up_eta_coeffs;
      get<::Tags::Modal<VectorTag<dim, 0>>>(
          package_data_lo_eta.modal_volume_data)
          .get(i) = lo_eta_coeffs;
      get<::Tags::Modal<VectorTag<dim, 0>>>(
          package_data_up_zeta.modal_volume_data)
          .get(i) = up_zeta_coeffs;
      get<::Tags::Modal<VectorTag<dim, 0>>>(
          package_data_lo_zeta.modal_volume_data)
          .get(i) = lo_zeta_coeffs;
    }

    krivodonova(&nodal_scalar_data_to_limit, &nodal_vector_data_to_limit,
                element, mesh, neighbor_data);
    to_nodal_coefficients(&expected, expected_coeffs, mesh);
    CHECK_ITERABLE_APPROX(get(nodal_scalar_data_to_limit), expected);
    for (size_t i = 0; i < dim; ++i) {
      CHECK_ITERABLE_APPROX(nodal_vector_data_to_limit.get(i), expected);
    }
  };

  // Map between 3D and 1D coefficients:
  // [(0,0,0), (1,0,0), (2,0,0), (0,1,0), (1,1,0), (2,1,0), (0,2,0), (1,2,0),
  //  (0,       1,       2,       3,       4,       5,       6,       7,
  //
  //  (2,2,0), (0,0,1), (1,0,1), (2,0,1), (0,1,1), (1,1,1), (2,1,1), (0,2,1),
  //   8,       9,       10,      11,      12,      13,      14,      15,
  //
  //  (1,2,1), (2,2,1), (0,0,2), (1,0,2), (2,0,2), (0,1,2), (1,1,2), (2,1,2),
  //   16,      17,      18,      19,      20,      21,      22,      23,
  //
  //  (0,2,2), (1,2,2), (2,2,2)]
  //   24,      25,      26)

  test_limiting_2_2_2_coefficient(helper);
  test_limiting_2_2_1_coefficient(helper);
  test_limiting_2_1_2_coefficient(helper);
  test_limiting_1_2_2_coefficient(helper);
  test_limiting_0_1_2_coefficient_permutations(helper);

  test_limiting_different_values_different_tensors();
}
}  // namespace test_3d
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.DG.SlopeLimiters.Krivodonova",
                  "[SlopeLimiters][Unit]") {
  test_1d::run();
  test_2d::run();
  test_3d::run();
}
}  // namespace SlopeLimiters
