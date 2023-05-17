// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <unordered_set>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Evolution/DgSubcell/SliceData.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/FiniteDifference/PartialDerivatives.hpp"
#include "NumericalAlgorithms/FiniteDifference/PartialDerivatives.tpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"

namespace {
template <size_t Dim>
void test(const gsl::not_null<std::mt19937*> generator,
          const gsl::not_null<std::uniform_real_distribution<>*> dist,
          const size_t points_per_dimension, const size_t fd_order) {
  CAPTURE(points_per_dimension);
  CAPTURE(fd_order);
  CAPTURE(Dim);
  const size_t max_degree = fd_order - 1;
  const size_t stencil_width = fd_order + 1;
  const size_t number_of_vars = 2;  // arbitrary, 2 is "cheap but not trivial"

  const Mesh<Dim> mesh{points_per_dimension, Spectral::Basis::FiniteDifference,
                       Spectral::Quadrature::CellCentered};
  auto logical_coords = logical_coordinates(mesh);
  // Make the logical coordinates different in each direction
  for (size_t i = 1; i < Dim; ++i) {
    logical_coords.get(i) += 4.0 * i;
  }

  // Compute polynomial on cell centers in FD cluster of points
  const auto set_polynomial = [max_degree](
                                  const gsl::not_null<DataVector*> var1_ptr,
                                  const gsl::not_null<DataVector*> var2_ptr,
                                  const auto& local_logical_coords) {
    *var1_ptr = 0.0;
    *var2_ptr = 100.0;  // some constant offset to distinguish the var values
    for (size_t degree = 1; degree <= max_degree; ++degree) {
      for (size_t i = 0; i < Dim; ++i) {
        *var1_ptr += pow(local_logical_coords.get(i), degree);
        *var2_ptr += pow(local_logical_coords.get(i), degree);
      }
    }
  };
  const auto set_polynomial_derivative =
      [max_degree](const gsl::not_null<std::array<DataVector, Dim>*> d_var1_ptr,
                   const gsl::not_null<std::array<DataVector, Dim>*> d_var2_ptr,
                   const auto& local_logical_coords) {
        for (size_t deriv_dim = 0; deriv_dim < Dim; ++deriv_dim) {
          gsl::at(*d_var1_ptr, deriv_dim) = 0.0;
          // constant deriv is zero
          gsl::at(*d_var2_ptr, deriv_dim) = 0.0;
          for (size_t degree = 1; degree <= max_degree; ++degree) {
            gsl::at(*d_var1_ptr, deriv_dim) +=
                degree * pow(local_logical_coords.get(deriv_dim), degree - 1);
            gsl::at(*d_var2_ptr, deriv_dim) +=
                degree * pow(local_logical_coords.get(deriv_dim), degree - 1);
          }
        }
      };

  DataVector volume_vars{mesh.number_of_grid_points() * number_of_vars, 0.0};
  DataVector var1(volume_vars.data(), mesh.number_of_grid_points());
  DataVector var2(volume_vars.data() + mesh.number_of_grid_points(),  // NOLINT
                  mesh.number_of_grid_points());
  set_polynomial(&var1, &var2, logical_coords);

  DataVector expected_deriv{Dim * volume_vars.size()};
  std::array<DataVector, Dim> expected_d_var1{};
  std::array<DataVector, Dim> expected_d_var2{};
  for (size_t i = 0; i < Dim; ++i) {
    gsl::at(expected_d_var1, i)
        .set_data_ref(&expected_deriv[i * volume_vars.size()],
                      mesh.number_of_grid_points());
    gsl::at(expected_d_var2, i)
        .set_data_ref(&expected_deriv[i * volume_vars.size() +
                                      mesh.number_of_grid_points()],
                      mesh.number_of_grid_points());
  }
  set_polynomial_derivative(&expected_d_var1, &expected_d_var2, logical_coords);

  // Compute the polynomial at the cell center for the neighbor data that we
  // "received".
  //
  // We do this by computing the solution in our entire neighbor, then using
  // slice_data to get the subset of points that are needed.
  DirectionMap<Dim, DataVector> neighbor_data{};
  for (const auto& direction : Direction<Dim>::all_directions()) {
    auto neighbor_logical_coords = logical_coords;
    neighbor_logical_coords.get(direction.dimension()) +=
        direction.sign() * 2.0;
    DataVector neighbor_vars{mesh.number_of_grid_points() * number_of_vars,
                             0.0};
    DataVector neighbor_var1(neighbor_vars.data(),
                             mesh.number_of_grid_points());
    DataVector neighbor_var2(
        neighbor_vars.data() + mesh.number_of_grid_points(),  // NOLINT
        mesh.number_of_grid_points());
    set_polynomial(&neighbor_var1, &neighbor_var2, neighbor_logical_coords);

    const auto sliced_data = evolution::dg::subcell::detail::slice_data_impl(
        gsl::make_span(neighbor_vars.data(), neighbor_vars.size()),
        mesh.extents(), (stencil_width - 1) / 2 + 1,
        std::unordered_set{direction.opposite()}, 0);
    CAPTURE((stencil_width - 1) / 2 + 1);
    REQUIRE(sliced_data.size() == 1);
    REQUIRE(sliced_data.contains(direction.opposite()));
    neighbor_data[direction] = sliced_data.at(direction.opposite());
    REQUIRE(neighbor_data.at(direction).size() ==
            number_of_vars * (fd_order / 2 + 1) *
                mesh.slice_away(0).number_of_grid_points());
  }

  // Note: reconstructed_num_pts assumes isotropic extents
  DataVector logical_derivative_buffer{volume_vars.size() * Dim};
  std::array<gsl::span<double>, Dim> logical_derivative_view{};
  for (size_t i = 0; i < Dim; ++i) {
    gsl::at(logical_derivative_view, i) = gsl::make_span(
        &logical_derivative_buffer[i * volume_vars.size()], volume_vars.size());
  }

  DirectionMap<Dim, gsl::span<const double>> ghost_cell_vars{};
  for (const auto& [direction, data] : neighbor_data) {
    ghost_cell_vars[direction] = gsl::make_span(data.data(), data.size());
  }

  ::fd::logical_partial_derivatives(
      make_not_null(&logical_derivative_view),
      gsl::make_span(volume_vars.data(), volume_vars.size()), ghost_cell_vars,
      mesh, number_of_vars, fd_order);

  // Scale to volume_vars since that sets the subtraction error threshold.
  Approx custom_approx = Approx::custom().epsilon(1.0e-14).scale(
      *std::max_element(volume_vars.begin(), volume_vars.end()));

  for (size_t i = 0; i < Dim; ++i) {
    CAPTURE(i);
    {
      CAPTURE(var1);
      const DataVector fd_d_var1(&gsl::at(logical_derivative_view, i)[0],
                                 mesh.number_of_grid_points());
      CHECK_ITERABLE_CUSTOM_APPROX(fd_d_var1, gsl::at(expected_d_var1, i),
                                   custom_approx);
    }
    {
      CAPTURE(var2);
      const DataVector fd_d_var2(
          &gsl::at(logical_derivative_view, i)[mesh.number_of_grid_points()],
          mesh.number_of_grid_points());
      CHECK_ITERABLE_CUSTOM_APPROX(fd_d_var2, gsl::at(expected_d_var2, i),
                                   custom_approx);
    }
  }

  // Test partial derivative with random Jacobian. We know we calculated the
  // logical partial derivatives correctly, just need to make sure we forward to
  // the other functions correctly.
  const auto inverse_jacobian = make_with_random_values<
      InverseJacobian<DataVector, Dim, Frame::ElementLogical, Frame::Inertial>>(
      generator, dist, DataVector{mesh.number_of_grid_points()});

  using derivative_tags = tmpl::list<Tags::TempScalar<0, DataVector>,
                                     Tags::TempScalar<1, DataVector>>;
  Variables<db::wrap_tags_in<Tags::deriv, derivative_tags, tmpl::size_t<Dim>,
                             Frame::Inertial>>
      partial_derivatives{mesh.number_of_grid_points()};
  ::fd::partial_derivatives<derivative_tags>(
      make_not_null(&partial_derivatives),
      gsl::make_span(volume_vars.data(), volume_vars.size()), ghost_cell_vars,
      mesh, number_of_vars, fd_order, inverse_jacobian);

  std::array<const double*, Dim> expected_logical_derivs_ptrs{};
  for (size_t i = 0; i < Dim; ++i) {
    gsl::at(expected_logical_derivs_ptrs, i) =
        gsl::at(expected_d_var1, i).data();
  }
  Variables<db::wrap_tags_in<Tags::deriv, derivative_tags, tmpl::size_t<Dim>,
                             Frame::Inertial>>
      expected_partial_derivatives{mesh.number_of_grid_points()};
  ::partial_derivatives_detail::partial_derivatives_impl<derivative_tags>(
      make_not_null(&expected_partial_derivatives),
      expected_logical_derivs_ptrs, inverse_jacobian);

  using d_var1_tag = Tags::deriv<Tags::TempScalar<0, DataVector>,
                                 tmpl::size_t<Dim>, Frame::Inertial>;
  using d_var2_tag = Tags::deriv<Tags::TempScalar<1, DataVector>,
                                 tmpl::size_t<Dim>, Frame::Inertial>;
  CHECK_ITERABLE_CUSTOM_APPROX(get<d_var1_tag>(partial_derivatives),
                               get<d_var1_tag>(expected_partial_derivatives),
                               custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(get<d_var2_tag>(partial_derivatives),
                               get<d_var2_tag>(expected_partial_derivatives),
                               custom_approx);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.FiniteDifference.PartialDerivatives",
                  "[Unit][NumericalAlgorithms]") {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist{-1.0, 1.0};
  for (const size_t fd_order : {2_st, 4_st, 6_st, 8_st}) {
    test<1>(make_not_null(&generator), make_not_null(&dist), fd_order + 2,
            fd_order);
    test<2>(make_not_null(&generator), make_not_null(&dist), fd_order + 2,
            fd_order);
    test<3>(make_not_null(&generator), make_not_null(&dist), fd_order + 2,
            fd_order);
  }
}
