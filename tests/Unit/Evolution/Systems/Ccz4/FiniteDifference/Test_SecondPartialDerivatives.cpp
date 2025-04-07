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
#include "Evolution/Systems/Ccz4/FiniteDifference/SecondPartialDerivatives.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/SecondPartialDerivatives.tpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"

namespace {
template <size_t Dim>
void test(const size_t points_per_dimension, const size_t fd_order) {
  // The following code assumes fd_order == 4 and Dim == 3
  ASSERT((Dim == 3) and (fd_order == 4),
         "Only 3 spatial dims and 4th-order fd is supported right now!");
  CAPTURE(points_per_dimension);
  CAPTURE(fd_order);
  CAPTURE(Dim);

  const size_t max_degree = fd_order - 1;
  const size_t number_of_vars = 2;

  const Mesh<Dim> mesh{points_per_dimension, Spectral::Basis::FiniteDifference,
                       Spectral::Quadrature::CellCentered};
  auto logical_coords = logical_coordinates(mesh);
  // Make the logical coordinates different in each direction
  for (size_t i = 0; i < Dim; ++i) {
    logical_coords.get(i) += 4.0 * static_cast<double>(i + 1);
  }

  // Set up a polynomial of max_degree on cell centers in FD cluster of points
  const auto set_polynomial = [max_degree](
                                  const gsl::not_null<DataVector*> var1_ptr,
                                  const gsl::not_null<DataVector*> var2_ptr,
                                  const auto& local_logical_coords) {
    *var1_ptr = 0.0;
    *var2_ptr = 100.0;  // some constant offset to distinguish the var values
    for (size_t degree_x = 0; degree_x <= max_degree; ++degree_x) {
      for (size_t degree_y = 0; degree_y <= max_degree; ++degree_y) {
        for (size_t degree_z = 0; degree_z <= max_degree; ++degree_z) {
          if (degree_x + degree_y + degree_z <= max_degree) {
            *var1_ptr += pow(local_logical_coords.get(0), degree_x) *
                         pow(local_logical_coords.get(1), degree_y) *
                         pow(local_logical_coords.get(2), degree_z);
            *var2_ptr += pow(local_logical_coords.get(0), degree_x) *
                         pow(local_logical_coords.get(1), degree_y) *
                         pow(local_logical_coords.get(2), degree_z);
          }
        }
      }
    }
  };

  // Compute the expected pure second derivatives of the polynomial
  const auto set_polynomial_pure_second_derivative =
      [](const gsl::not_null<std::array<DataVector, Dim>*>
             pure_second_d_var1_ptr,
         const gsl::not_null<std::array<DataVector, Dim>*>
             pure_second_d_var2_ptr,
         const auto& local_logical_coords) {
        for (size_t deriv_dim = 0; deriv_dim < Dim; ++deriv_dim) {
          gsl::at(*pure_second_d_var1_ptr, deriv_dim) =
              2 + 2 * local_logical_coords.get(0) +
              2 * local_logical_coords.get(1) + 2 * local_logical_coords.get(2);
          gsl::at(*pure_second_d_var1_ptr, deriv_dim) +=
              4 * local_logical_coords.get(deriv_dim);

          gsl::at(*pure_second_d_var2_ptr, deriv_dim) =
              2 + 2 * local_logical_coords.get(0) +
              2 * local_logical_coords.get(1) + 2 * local_logical_coords.get(2);
          gsl::at(*pure_second_d_var2_ptr, deriv_dim) +=
              4 * local_logical_coords.get(deriv_dim);
        }
      };

  // Compute the expected mixed second derivatives of the polynomial
  // deriv_dim = 0 -- xy derivative; 1 -- yz derivative; 2 -- xz derivative
  const auto set_polynomial_mixed_second_derivative =
      [](const gsl::not_null<std::array<DataVector, Dim>*>
             mixed_second_d_var1_ptr,
         const gsl::not_null<std::array<DataVector, Dim>*>
             mixed_second_d_var2_ptr,
         const auto& local_logical_coords) {
        for (size_t deriv_dim = 0; deriv_dim < Dim; ++deriv_dim) {
          gsl::at(*mixed_second_d_var1_ptr, deriv_dim) =
              1 + local_logical_coords.get(0) + local_logical_coords.get(1) +
              local_logical_coords.get(2);
          gsl::at(*mixed_second_d_var1_ptr, deriv_dim) +=
              local_logical_coords.get(deriv_dim % Dim) +
              local_logical_coords.get((deriv_dim + 1) % Dim);
          gsl::at(*mixed_second_d_var2_ptr, deriv_dim) =
              1 + local_logical_coords.get(0) + local_logical_coords.get(1) +
              local_logical_coords.get(2);
          gsl::at(*mixed_second_d_var2_ptr, deriv_dim) +=
              local_logical_coords.get(deriv_dim % Dim) +
              local_logical_coords.get((deriv_dim + 1) % Dim);
        }
      };

  DataVector volume_vars{mesh.number_of_grid_points() * number_of_vars, 0.0};
  DataVector var1(volume_vars.data(), mesh.number_of_grid_points());
  DataVector var2(volume_vars.data() + mesh.number_of_grid_points(),
                  mesh.number_of_grid_points());
  set_polynomial(&var1, &var2, logical_coords);

  DataVector expected_pure_second_deriv{Dim * volume_vars.size()};
  std::array<DataVector, Dim> expected_pure_second_d_var1{};
  std::array<DataVector, Dim> expected_pure_second_d_var2{};
  for (size_t i = 0; i < Dim; ++i) {
    gsl::at(expected_pure_second_d_var1, i)
        .set_data_ref(&expected_pure_second_deriv[i * volume_vars.size()],
                      mesh.number_of_grid_points());
    gsl::at(expected_pure_second_d_var2, i)
        .set_data_ref(&expected_pure_second_deriv[i * volume_vars.size() +
                                                  mesh.number_of_grid_points()],
                      mesh.number_of_grid_points());
  }
  set_polynomial_pure_second_derivative(&expected_pure_second_d_var1,
                                        &expected_pure_second_d_var2,
                                        logical_coords);

  DataVector expected_mixed_second_deriv{Dim * volume_vars.size()};
  std::array<DataVector, Dim> expected_mixed_second_d_var1{};
  std::array<DataVector, Dim> expected_mixed_second_d_var2{};
  for (size_t i = 0; i < Dim; ++i) {
    gsl::at(expected_mixed_second_d_var1, i)
        .set_data_ref(&expected_mixed_second_deriv[i * volume_vars.size()],
                      mesh.number_of_grid_points());
    gsl::at(expected_mixed_second_d_var2, i)
        .set_data_ref(
            &expected_mixed_second_deriv[i * volume_vars.size() +
                                         mesh.number_of_grid_points()],
            mesh.number_of_grid_points());
  }
  set_polynomial_mixed_second_derivative(&expected_mixed_second_d_var1,
                                         &expected_mixed_second_d_var2,
                                         logical_coords);

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
        neighbor_vars.data() + mesh.number_of_grid_points(),
        mesh.number_of_grid_points());
    set_polynomial(&neighbor_var1, &neighbor_var2, neighbor_logical_coords);

    // We need two ghost points for 4th order
    const size_t num_of_ghost_pts = 2;
    const auto sliced_data = evolution::dg::subcell::detail::slice_data_impl(
        gsl::make_span(neighbor_vars.data(), neighbor_vars.size()),
        mesh.extents(), num_of_ghost_pts,
        std::unordered_set{direction.opposite()}, 0, {});
    REQUIRE(sliced_data.size() == 1);
    REQUIRE(sliced_data.contains(direction.opposite()));
    neighbor_data[direction] = sliced_data.at(direction.opposite());
    REQUIRE(neighbor_data.at(direction).size() ==
            number_of_vars * num_of_ghost_pts *
                mesh.slice_away(0).number_of_grid_points());
  }

  // Note: reconstructed_num_pts assumes isotropic extents
  DataVector pure_second_logical_derivative_buffer{volume_vars.size() * Dim};
  std::array<gsl::span<double>, Dim> pure_second_logical_derivative_view{};
  for (size_t i = 0; i < Dim; ++i) {
    gsl::at(pure_second_logical_derivative_view, i) = gsl::make_span(
        &pure_second_logical_derivative_buffer[i * volume_vars.size()],
        volume_vars.size());
  }

  DataVector mixed_second_logical_derivative_buffer{volume_vars.size() * Dim};
  std::array<gsl::span<double>, Dim> mixed_second_logical_derivative_view{};
  for (size_t i = 0; i < Dim; ++i) {
    gsl::at(mixed_second_logical_derivative_view, i) = gsl::make_span(
        &mixed_second_logical_derivative_buffer[i * volume_vars.size()],
        volume_vars.size());
  }

  DirectionMap<Dim, gsl::span<const double>> ghost_cell_vars{};
  for (const auto& [direction, data] : neighbor_data) {
    ghost_cell_vars[direction] = gsl::make_span(data.data(), data.size());
  }

  ::Ccz4::fd::second_logical_partial_derivatives(
      make_not_null(&pure_second_logical_derivative_view),
      make_not_null(&mixed_second_logical_derivative_view),
      gsl::make_span(volume_vars.data(), volume_vars.size()), ghost_cell_vars,
      mesh, number_of_vars, fd_order);

  // Scale to volume_vars since that sets the subtraction error threshold.
  Approx custom_approx = Approx::custom().epsilon(1.0e-12).scale(
      *std::max_element(volume_vars.begin(), volume_vars.end()));

  for (size_t i = 0; i < Dim; ++i) {
    CAPTURE(i);
    {
      CAPTURE(logical_coords.get(0));
      CAPTURE(var1);
      const DataVector fd_pure_second_d_var1(
          &gsl::at(pure_second_logical_derivative_view, i)[0],
          mesh.number_of_grid_points());
      CHECK_ITERABLE_CUSTOM_APPROX(fd_pure_second_d_var1,
                                   gsl::at(expected_pure_second_d_var1, i),
                                   custom_approx);
    }
    {
      CAPTURE(var2);
      const DataVector fd_pure_second_d_var2(
          &gsl::at(pure_second_logical_derivative_view,
                   i)[mesh.number_of_grid_points()],
          mesh.number_of_grid_points());
      CHECK_ITERABLE_CUSTOM_APPROX(fd_pure_second_d_var2,
                                   gsl::at(expected_pure_second_d_var2, i),
                                   custom_approx);
    }
    {
      CAPTURE(var1);
      const DataVector fd_mixed_second_d_var1(
          &gsl::at(mixed_second_logical_derivative_view, i)[0],
          mesh.number_of_grid_points());
      CHECK_ITERABLE_CUSTOM_APPROX(fd_mixed_second_d_var1,
                                   gsl::at(expected_mixed_second_d_var1, i),
                                   custom_approx);
    }
    {
      CAPTURE(var2);
      const DataVector fd_mixed_second_d_var2(
          &gsl::at(mixed_second_logical_derivative_view,
                   i)[mesh.number_of_grid_points()],
          mesh.number_of_grid_points());
      CHECK_ITERABLE_CUSTOM_APPROX(fd_mixed_second_d_var2,
                                   gsl::at(expected_mixed_second_d_var2, i),
                                   custom_approx);
    }
  }

  // Test second partial derivatives with Jacobian.
  // We use inertial coords x = \xi^2, y = \eta, and z = \xi + \zeta
  // We also change the first var to a tensor for generality
  InverseJacobian<DataVector, Dim, Frame::ElementLogical, Frame::Inertial>
      inverse_jacobian(mesh.number_of_grid_points(), 0.0);
  inverse_jacobian.get(0, 0) = 0.5 / logical_coords.get(0);
  inverse_jacobian.get(1, 1) = 1.0;
  inverse_jacobian.get(2, 0) = -0.5 / logical_coords.get(0);
  inverse_jacobian.get(2, 2) = 1.0;

  using derivative_tags =
      tmpl::list<Tags::Tempi<0, Dim, Frame::Inertial, DataVector>,
                 Tags::TempScalar<1, DataVector>>;

  Variables<db::wrap_tags_in<Tags::second_deriv, derivative_tags,
                             tmpl::size_t<Dim>, Frame::Inertial>>
      second_partial_derivatives{mesh.number_of_grid_points()};

  const size_t number_of_independent_components =
      Variables<derivative_tags>::number_of_independent_components;

  DataVector volume_vars_for_tensor(
      mesh.number_of_grid_points() * number_of_independent_components, 0.0);
  DataVector tensor_var_1(volume_vars_for_tensor.data(),
                          mesh.number_of_grid_points());
  DataVector tensor_var_2(
      volume_vars_for_tensor.data() + mesh.number_of_grid_points(),
      mesh.number_of_grid_points());
  DataVector tensor_var_3(
      volume_vars_for_tensor.data() + 2 * mesh.number_of_grid_points(),
      mesh.number_of_grid_points());
  DataVector scalar_var(
      volume_vars_for_tensor.data() + Dim * mesh.number_of_grid_points(),
      mesh.number_of_grid_points());

  set_polynomial(&tensor_var_1, &tensor_var_2, logical_coords);
  set_polynomial(&tensor_var_3, &scalar_var, logical_coords);

  // Update ghost data as we include tensor variable
  for (const auto& direction : Direction<Dim>::all_directions()) {
    auto neighbor_logical_coords = logical_coords;
    neighbor_logical_coords.get(direction.dimension()) +=
        direction.sign() * 2.0;
    DataVector neighbor_vars{
        mesh.number_of_grid_points() * number_of_independent_components, 0.0};
    DataVector neighbor_tensor_var1(neighbor_vars.data(),
                                    mesh.number_of_grid_points());
    DataVector neighbor_tensor_var2(
        neighbor_vars.data() + mesh.number_of_grid_points(),
        mesh.number_of_grid_points());
    DataVector neighbor_tensor_var3(
        neighbor_vars.data() + 2 * mesh.number_of_grid_points(),
        mesh.number_of_grid_points());
    DataVector neighbor_scalar_var(
        neighbor_vars.data() + Dim * mesh.number_of_grid_points(),  // NOLINT
        mesh.number_of_grid_points());
    set_polynomial(&neighbor_tensor_var1, &neighbor_tensor_var2,
                   neighbor_logical_coords);
    set_polynomial(&neighbor_tensor_var3, &neighbor_scalar_var,
                   neighbor_logical_coords);

    // We need two ghost points for 4th order
    const size_t num_of_ghost_pts = 2;
    const auto sliced_data = evolution::dg::subcell::detail::slice_data_impl(
        gsl::make_span(neighbor_vars.data(), neighbor_vars.size()),
        mesh.extents(), num_of_ghost_pts,
        std::unordered_set{direction.opposite()}, 0, {});
    REQUIRE(sliced_data.size() == 1);
    REQUIRE(sliced_data.contains(direction.opposite()));
    neighbor_data[direction] = sliced_data.at(direction.opposite());
    REQUIRE(neighbor_data.at(direction).size() ==
            number_of_independent_components * num_of_ghost_pts *
                mesh.slice_away(0).number_of_grid_points());
  }

  for (const auto& [direction, data] : neighbor_data) {
    ghost_cell_vars[direction] = gsl::make_span(data.data(), data.size());
  }

  ::Ccz4::fd::second_partial_derivatives<derivative_tags>(
      make_not_null(&second_partial_derivatives),
      gsl::make_span(volume_vars_for_tensor.data(),
                     volume_vars_for_tensor.size()),
      ghost_cell_vars, mesh, number_of_independent_components, 4,
      inverse_jacobian);

  Variables<db::wrap_tags_in<Tags::second_deriv, derivative_tags,
                             tmpl::size_t<Dim>, Frame::Inertial>>
      expected_second_partial_derivatives{mesh.number_of_grid_points()};

  // transform the expected derivs to inertial coords
  gsl::at(expected_pure_second_d_var1, 0) =
      (1 + 2 * logical_coords.get(0) + logical_coords.get(1) +
       2 * logical_coords.get(2)) /
      (2 * square(logical_coords.get(0)));
  gsl::at(expected_pure_second_d_var1, 1) =
      2 * (1 + logical_coords.get(0) + 3 * logical_coords.get(1) +
           logical_coords.get(2));
  gsl::at(expected_pure_second_d_var1, 2) =
      2 * (1 + logical_coords.get(0) + logical_coords.get(1) +
           3 * logical_coords.get(2));
  gsl::at(expected_pure_second_d_var2, 0) =
      (1 + 2 * logical_coords.get(0) + logical_coords.get(1) +
       2 * logical_coords.get(2)) /
      (2 * square(logical_coords.get(0)));
  gsl::at(expected_pure_second_d_var2, 1) =
      2 * (1 + logical_coords.get(0) + 3 * logical_coords.get(1) +
           logical_coords.get(2));
  gsl::at(expected_pure_second_d_var2, 2) =
      2 * (1 + logical_coords.get(0) + logical_coords.get(1) +
           3 * logical_coords.get(2));

  gsl::at(expected_mixed_second_d_var1, 0) =
      (logical_coords.get(0) - logical_coords.get(2)) /
      (2 * logical_coords.get(0));
  gsl::at(expected_mixed_second_d_var1, 1) = 1 + logical_coords.get(0) +
                                             2 * logical_coords.get(1) +
                                             2 * logical_coords.get(2);
  gsl::at(expected_mixed_second_d_var1, 2) =
      -(1 + logical_coords.get(1) + 4 * logical_coords.get(2)) /
      (2 * logical_coords.get(0));
  gsl::at(expected_mixed_second_d_var2, 0) =
      (logical_coords.get(0) - logical_coords.get(2)) /
      (2 * logical_coords.get(0));
  gsl::at(expected_mixed_second_d_var2, 1) = 1 + logical_coords.get(0) +
                                             2 * logical_coords.get(1) +
                                             2 * logical_coords.get(2);
  gsl::at(expected_mixed_second_d_var2, 2) =
      -(1 + logical_coords.get(1) + 4 * logical_coords.get(2)) /
      (2 * logical_coords.get(0));

  // putting the expected pure and mixed second derivs into the variable
  // expected_second_partial_derivatives
  using second_d_var1_tag =
      Tags::second_deriv<Tags::Tempi<0, Dim, Frame::Inertial, DataVector>,
                         tmpl::size_t<Dim>, Frame::Inertial>;
  using second_d_var2_tag =
      Tags::second_deriv<Tags::TempScalar<1, DataVector>, tmpl::size_t<Dim>,
                         Frame::Inertial>;
  for (size_t deriv_index = 0; deriv_index < Dim; deriv_index++) {
    for (size_t i = 0; i < Dim; ++i) {
      (get<second_d_var1_tag>(expected_second_partial_derivatives))
          .get(deriv_index, deriv_index, i) =
          gsl::at(expected_pure_second_d_var1, deriv_index);
      (get<second_d_var1_tag>(expected_second_partial_derivatives))
          .get(deriv_index, (deriv_index + 1) % Dim, i) =
          gsl::at(expected_mixed_second_d_var1, deriv_index);
    }
    (get<second_d_var2_tag>(expected_second_partial_derivatives))
        .get(deriv_index, deriv_index) =
        gsl::at(expected_pure_second_d_var2, deriv_index);

    (get<second_d_var2_tag>(expected_second_partial_derivatives))
        .get(deriv_index, (deriv_index + 1) % Dim) =
        gsl::at(expected_mixed_second_d_var2, deriv_index);
  }
  {
    CHECK_ITERABLE_CUSTOM_APPROX(
        get<second_d_var1_tag>(second_partial_derivatives),
        get<second_d_var1_tag>(expected_second_partial_derivatives),
        custom_approx);
  }
  {
    CHECK_ITERABLE_CUSTOM_APPROX(
        get<second_d_var2_tag>(second_partial_derivatives),
        get<second_d_var2_tag>(expected_second_partial_derivatives),
        custom_approx);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.FiniteDifference.SecondPartialDerivatives",
                  "[Unit][NumericalAlgorithms]") {
  const size_t Dim = 3;
  const size_t points_per_dimension = 5;
  const size_t fd_deriv_order = 4;
  test<Dim>(points_per_dimension, fd_deriv_order);
}
