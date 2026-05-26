// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <unordered_set>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Evolution/DgSubcell/SliceData.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/FiniteDifference/PartialDerivatives.hpp"
#include "NumericalAlgorithms/FiniteDifference/PartialDerivatives.tpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
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
        std::unordered_set{direction.opposite()}, 0, {});
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
  ::partial_derivatives_detail::partial_derivatives_impl(
      make_not_null(&expected_partial_derivatives),
      expected_logical_derivs_ptrs,
      Variables<derivative_tags>::number_of_independent_components,
      inverse_jacobian);

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

  // Test the partial derivative "chooser" works correctly when we pass an
  // unused inertial_coords
  const tnsr::I<DataVector, Dim, Frame::Inertial> inertial_coords{};
  using var_tags = tmpl::list<Tags::TempScalar<0, DataVector>,
                              Tags::TempScalar<1, DataVector>>;
  Variables<var_tags> volume_variables{mesh.number_of_grid_points()};
  get(get<Tags::TempScalar<0>>(volume_variables)) = var1;
  get(get<Tags::TempScalar<1>>(volume_variables)) = var2;
  ::fd::partial_derivatives<var_tags>(
      make_not_null(&partial_derivatives), volume_variables, ghost_cell_vars,
      mesh, Variables<var_tags>::number_of_independent_components, fd_order,
      inverse_jacobian, inertial_coords);
  CHECK_VARIABLES_CUSTOM_APPROX(partial_derivatives,
                                expected_partial_derivatives, custom_approx);
}

template <bool Spherical>
void test_cartoon(const size_t points_per_dimension, const size_t fd_order) {
  INFO("Testing FD partial derivatives with Cartoon bases");
  CAPTURE(points_per_dimension);
  CAPTURE(fd_order);
  CAPTURE(Spherical);
  constexpr size_t CompDim = Spherical ? 1 : 2;
  const size_t max_degree = fd_order - 1;
  const size_t stencil_width = fd_order + 1;

  Mesh<3> mesh;
  tnsr::I<DataVector, 3, Frame::Inertial> inertial_coords;
  InverseJacobian<DataVector, 3, Frame::ElementLogical, Frame::Inertial>
      inv_jacobian;

  using Affine = domain::CoordinateMaps::Affine;
  using Identity1D = domain::CoordinateMaps::Identity<1>;
  const Identity1D identity_cartoon_map;

  if constexpr (Spherical) {
    mesh = Mesh<3>{{points_per_dimension, 1, 1},
                   {Spectral::Basis::FiniteDifference, Spectral::Basis::Cartoon,
                    Spectral::Basis::Cartoon},
                   {Spectral::Quadrature::CellCentered,
                    Spectral::Quadrature::SphericalSymmetry,
                    Spectral::Quadrature::SphericalSymmetry}};
    using Cartoon_map_combination =
        domain::CoordinateMaps::ProductOf3Maps<Affine, Identity1D, Identity1D>;
    const domain::CoordinateMap<Frame::ElementLogical, Frame::Inertial,
                                Cartoon_map_combination>
        map{{Affine{-1.0, 1.0, 1.5, 2.2}, identity_cartoon_map,
             identity_cartoon_map}};
    inv_jacobian = map.inv_jacobian(logical_coordinates(mesh));
    inertial_coords = map(logical_coordinates(mesh));
  } else {
    mesh = Mesh<3>{
        {points_per_dimension, points_per_dimension, 1},
        {Spectral::Basis::FiniteDifference, Spectral::Basis::FiniteDifference,
         Spectral::Basis::Cartoon},
        {Spectral::Quadrature::CellCentered, Spectral::Quadrature::CellCentered,
         Spectral::Quadrature::AxialSymmetry}};
    using Cartoon_map_combination =
        domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Identity1D>;
    const domain::CoordinateMap<Frame::ElementLogical, Frame::Inertial,
                                Cartoon_map_combination>
        map{{Affine{-1.0, 1.0, 0.4, 0.7}, Affine{-1.0, 1.0, -0.5, 3.2},
             identity_cartoon_map}};
    inv_jacobian = map.inv_jacobian(logical_coordinates(mesh));
    inertial_coords = map(logical_coordinates(mesh));
  }

  const auto cartoon_func =
      [max_degree](
          const tnsr::I<DataVector, 3, Frame::Inertial>& coords) -> DataVector {
    if constexpr (Spherical) {
      // a radial function, f(r) = f(x) because the computational domain is the
      // x axis
      return 0.01 * pow(get<0>(coords), max_degree) +
             0.3 * pow(get<0>(coords), max_degree - 1) - 2.0 * get<0>(coords) -
             1.5;
    } else {
      // an axially symmetric function about the y axis,
      // f(\sqrt{x^2 + z^2}, y) = f(x, y) because the computational domain is
      // the x-y plane
      return pow(get<1>(coords), max_degree) +
             pow(get<0>(coords), max_degree) * get<1>(coords);
    }
  };

  const auto cartoon_dfunc =
      [max_degree](
          size_t deriv_index,
          const tnsr::I<DataVector, 3, Frame::Inertial>& coords) -> DataVector {
    if constexpr (Spherical) {
      if (deriv_index == 0) {
        return 0.01 * static_cast<double>(max_degree) *
                   pow(get<0>(coords), max_degree - 1) +
               0.3 * static_cast<double>(max_degree - 1) *
                   pow(get<0>(coords), max_degree - 2) -
               2.0;
      } else {
        return {get<0>(coords).size(), 0.0};
      }
    } else {
      if (deriv_index == 0) {
        return max_degree * pow(get<0>(coords), max_degree - 1) *
               get<1>(coords);
      } else if (deriv_index == 1) {
        return max_degree * pow(get<1>(coords), max_degree - 1) +
               pow(get<0>(coords), max_degree);
      } else {
        return {get<0>(coords).size(), 0.0};
      }
    }
  };

  using VarTags = tmpl::list<::Tags::TempScalar<0>, ::Tags::TempA<0, 3>>;
  const size_t number_var_components =
      Variables<VarTags>::number_of_independent_components;
  const size_t num_pts = get<0>(inertial_coords).size();
  Variables<VarTags> volume_vars{mesh.number_of_grid_points()};

  using d_VarTags =
      db::wrap_tags_in<Tags::deriv, VarTags, tmpl::size_t<3>, Frame::Inertial>;
  Variables<d_VarTags> expected_deriv{mesh.number_of_grid_points()};

  auto& scalar = get<::Tags::TempScalar<0>>(volume_vars);
  auto& d_scalar = get<tmpl::front<d_VarTags>>(expected_deriv);
  get(scalar) = cartoon_func(inertial_coords);
  for (size_t i = 0; i < index_dim<0>(d_scalar); ++i) {
    d_scalar.get(i) = cartoon_dfunc(i, inertial_coords);
  }

  auto& vector = get<::Tags::TempA<0, 3>>(volume_vars);
  auto& d_vector = get<tmpl::back<d_VarTags>>(expected_deriv);
  if constexpr (Spherical) {
    // spherical case, vector, using x^a
    get<0>(vector) = cartoon_func(inertial_coords);
    for (size_t a = 1; a < index_dim<0>(vector); ++a) {
      vector.get(a) =
          inertial_coords.get(a - 1) * cartoon_func(inertial_coords);
    }
    for (size_t i = 0; i < index_dim<0>(d_vector); ++i) {
      for (size_t a = 0; a < index_dim<1>(d_vector); ++a) {
        if ((i + 1) == a) {
          d_vector.get(i, a) = cartoon_func(inertial_coords);
        } else {
          d_vector.get(i, a) = DataVector(num_pts, 0.0);
        }
        d_vector.get(i, a) +=
            (a == 0 ? DataVector(num_pts, 1.0) : inertial_coords.get(a - 1)) *
            cartoon_dfunc(i, inertial_coords);
      }
    }
  } else {
    // axial case, vector, using x^a = (0, -z, 0, x) (pure rotation)
    get<0>(vector) = DataVector(num_pts, 0.0);
    get<1>(vector) =
        -1.0 * get<2>(inertial_coords) * cartoon_func(inertial_coords);
    get<2>(vector) = DataVector(num_pts, 0.0);
    get<3>(vector) = get<0>(inertial_coords) * cartoon_func(inertial_coords);

    for (size_t i = 0; i < d_vector.size(); ++i) {
      d_vector[i] = DataVector(num_pts, 0.0);
    }
    d_vector.get(2, 1) =
        -1.0 * cartoon_func(inertial_coords) +
        -1.0 * get<2>(inertial_coords) * cartoon_dfunc(2, inertial_coords);
    d_vector.get(1, 3) =
        get<0>(inertial_coords) * cartoon_dfunc(1, inertial_coords);
    d_vector.get(0, 3) =
        cartoon_func(inertial_coords) +
        get<0>(inertial_coords) * cartoon_dfunc(0, inertial_coords);
  }

  DirectionMap<3, DataVector> neighbor_data{};
  for (const auto& direction : Direction<CompDim>::all_directions()) {
    auto neighbor_inertial_coords = inertial_coords;
    auto& neighbor_coords_current_dim =
        neighbor_inertial_coords.get(direction.dimension());
    const double min_bounds = *std::min_element(
        neighbor_coords_current_dim.begin(), neighbor_coords_current_dim.end());
    const double max_bounds = *std::max_element(
        neighbor_coords_current_dim.begin(), neighbor_coords_current_dim.end());
    const double delta_bounds = max_bounds - min_bounds;
    const double delta_i =
        delta_bounds / static_cast<double>(points_per_dimension - 1);
    neighbor_coords_current_dim += direction.sign() * (delta_bounds + delta_i);
    DataVector neighbor_vars{
        mesh.number_of_grid_points() * number_var_components, 0.0};
    Scalar<DataVector> neighbor_var1(neighbor_vars.data(),
                                     mesh.number_of_grid_points());
    DataVector neighbor_var2_0(
        neighbor_vars.data() + 1 * mesh.number_of_grid_points(),
        mesh.number_of_grid_points());
    DataVector neighbor_var2_1(
        neighbor_vars.data() + 2 * mesh.number_of_grid_points(),
        mesh.number_of_grid_points());
    DataVector neighbor_var2_2(
        neighbor_vars.data() + 3 * mesh.number_of_grid_points(),
        mesh.number_of_grid_points());
    DataVector neighbor_var2_3(
        neighbor_vars.data() + 4 * mesh.number_of_grid_points(),
        mesh.number_of_grid_points());

    get(neighbor_var1) = cartoon_func(neighbor_inertial_coords);

    if constexpr (Spherical) {
      // spherical case, vector/one form, using x^a
      neighbor_var2_0 = cartoon_func(neighbor_inertial_coords);
      neighbor_var2_1 = get<0>(neighbor_inertial_coords) *
                        cartoon_func(neighbor_inertial_coords);
      neighbor_var2_2 = get<1>(neighbor_inertial_coords) *
                        cartoon_func(neighbor_inertial_coords);
      neighbor_var2_3 = get<2>(neighbor_inertial_coords) *
                        cartoon_func(neighbor_inertial_coords);
    } else {
      // axial case, vector/one form, using x^a = (0, -z, 0, x) (pure
      // rotation)
      neighbor_var2_0 = DataVector(num_pts, 0.0);
      neighbor_var2_1 = -1.0 * get<2>(neighbor_inertial_coords) *
                        cartoon_func(neighbor_inertial_coords);
      neighbor_var2_2 = DataVector(num_pts, 0.0);
      neighbor_var2_3 = get<0>(neighbor_inertial_coords) *
                        cartoon_func(neighbor_inertial_coords);
    }

    Index<CompDim> mesh_extents;
    if constexpr (Spherical) {
      mesh_extents = mesh.slice_through(0).extents();
    } else {
      mesh_extents = mesh.slice_through(0, 1).extents();
    }
    const auto sliced_data = evolution::dg::subcell::detail::slice_data_impl(
        gsl::make_span(neighbor_vars.data(), neighbor_vars.size()),
        mesh_extents, (stencil_width - 1) / 2 + 1,
        std::unordered_set{direction.opposite()}, 0, {});
    CAPTURE((stencil_width - 1) / 2 + 1);
    REQUIRE(sliced_data.size() == 1);
    REQUIRE(sliced_data.contains(direction.opposite()));
    const Direction<3> direction_3D{direction.dimension(), direction.side()};
    neighbor_data[direction_3D] = sliced_data.at(direction.opposite());
    REQUIRE(neighbor_data.at(direction_3D).size() ==
            number_var_components * (fd_order / 2 + 1) *
                mesh.slice_away(0).number_of_grid_points());
  }

  DirectionMap<3, gsl::span<const double>> ghost_cell_vars{};
  for (const auto& [direction, data] : neighbor_data) {
    ghost_cell_vars[direction] = gsl::make_span(data.data(), data.size());
  }

  Variables<d_VarTags> deriv{mesh.number_of_grid_points()};
  // testing the "chooser" as well as the underlying
  // `cartoon_partial_derivatives()`
  ::fd::partial_derivatives<VarTags>(
      make_not_null(&deriv), volume_vars, ghost_cell_vars, mesh,
      number_var_components, fd_order, inv_jacobian, inertial_coords);

  const Approx local_approx = Approx::custom().epsilon(1.0e-10).scale(1.0);
  CHECK_VARIABLES_CUSTOM_APPROX(deriv, expected_deriv, local_approx);
}

void test_asserts_and_errors() {
  using var_tags = tmpl::list<Tags::TempScalar<0, DataVector>,
                              Tags::TempScalar<1, DataVector>>;
  constexpr size_t fd_order = 4;
  constexpr size_t pts = 6;  // fd_order + 2

#ifdef SPECTRE_DEBUG
  using derivative_tags = tmpl::list<Tags::TempScalar<0, DataVector>,
                                     Tags::TempScalar<1, DataVector>>;
  {
    INFO("Wrong output size");
    const Mesh<2> mesh{pts, Spectral::Basis::FiniteDifference,
                       Spectral::Quadrature::CellCentered};
    const size_t num_pts = mesh.number_of_grid_points();
    const size_t num_vars = 2;
    DataVector volume_vars_dv{num_pts * num_vars, 0.0};
    const auto volume_vars_span =
        gsl::make_span(volume_vars_dv.data(), volume_vars_dv.size());

    // Create ghost cell data for all directions
    const size_t ghost_pts_per_face =
        (fd_order / 2 + 1) * mesh.slice_away(0).number_of_grid_points();
    DirectionMap<2, gsl::span<const double>> ghost_cell_vars{};
    DataVector ghost_dv{ghost_pts_per_face * num_vars, 0.0};
    for (const auto& dir : Direction<2>::all_directions()) {
      ghost_cell_vars[dir] = gsl::make_span(ghost_dv.data(), ghost_dv.size());
    }

    const InverseJacobian<DataVector, 2, Frame::ElementLogical, Frame::Inertial>
        inv_jac{num_pts, 0.0};

    // Allocate with wrong size (should be Dim * volume_vars.size())
    Variables<db::wrap_tags_in<Tags::deriv, derivative_tags, tmpl::size_t<2>,
                               Frame::Inertial>>
        d_vars_wrong{num_pts - 1};  // wrong number of grid points
    CHECK_THROWS_WITH((::fd::partial_derivatives<derivative_tags>(
                          make_not_null(&d_vars_wrong), volume_vars_span,
                          ghost_cell_vars, mesh, num_vars, fd_order, inv_jac)),
                      Catch::Matchers::ContainsSubstring(
                          "The partial derivatives Variables must have size"));
  }
  {
    INFO("Invalid Cartoon quadrature combination");
    // Build a 3D mesh with Cartoon in dim 2 but wrong quadrature
    const Mesh<3> mesh_wrong_quad{
        {pts, pts, 1},
        {Spectral::Basis::FiniteDifference, Spectral::Basis::FiniteDifference,
         Spectral::Basis::Cartoon},
        // Wrong: SphericalSymmetry instead of AxialSymmetry for 2 comp dims
        {Spectral::Quadrature::CellCentered, Spectral::Quadrature::CellCentered,
         Spectral::Quadrature::SphericalSymmetry}};
    const size_t num_pts = mesh_wrong_quad.number_of_grid_points();
    const size_t num_vars =
        Variables<var_tags>::number_of_independent_components;

    using Affine = domain::CoordinateMaps::Affine;
    using Identity1D = domain::CoordinateMaps::Identity<1>;
    using Cartoon_map_combination =
        domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Identity1D>;
    const domain::CoordinateMap<Frame::ElementLogical, Frame::Inertial,
                                Cartoon_map_combination>
        map{{Affine{-1.0, 1.0, 0.4, 0.7}, Affine{-1.0, 1.0, -0.5, 3.2},
             Identity1D{}}};
    const auto inv_jac = map.inv_jacobian(logical_coordinates(mesh_wrong_quad));
    const auto inertial_coords = map(logical_coordinates(mesh_wrong_quad));

    const Variables<var_tags> volume_vars{num_pts};
    Variables<db::wrap_tags_in<Tags::deriv, var_tags, tmpl::size_t<3>,
                               Frame::Inertial>>
        d_vars{num_pts};

    const size_t ghost_pts_per_face =
        (fd_order / 2 + 1) *
        mesh_wrong_quad.slice_away(0).number_of_grid_points();
    DirectionMap<3, gsl::span<const double>> ghost_cell_vars{};
    DataVector ghost_dv{ghost_pts_per_face * num_vars, 0.0};
    for (const auto& dir : Direction<2>::all_directions()) {
      ghost_cell_vars[Direction<3>{dir.dimension(), dir.side()}] =
          gsl::make_span(ghost_dv.data(), ghost_dv.size());
    }

    CHECK_THROWS_WITH(
        (::fd::cartoon_partial_derivatives<var_tags>(
            make_not_null(&d_vars), volume_vars, ghost_cell_vars,
            mesh_wrong_quad, num_vars, fd_order, inv_jac, inertial_coords)),
        Catch::Matchers::ContainsSubstring("Invalid Quadrature combinations"));
  }
  {
    INFO("cartoon impl, wrong deriv output size");
    const Mesh<3> mesh{
        {pts, pts, 1},
        {Spectral::Basis::FiniteDifference, Spectral::Basis::FiniteDifference,
         Spectral::Basis::Cartoon},
        {Spectral::Quadrature::CellCentered, Spectral::Quadrature::CellCentered,
         Spectral::Quadrature::AxialSymmetry}};
    const size_t num_pts = mesh.number_of_grid_points();
    const size_t num_vars =
        Variables<var_tags>::number_of_independent_components;

    using Affine = domain::CoordinateMaps::Affine;
    using Identity1D = domain::CoordinateMaps::Identity<1>;
    using Cartoon_map_combination =
        domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Identity1D>;
    const domain::CoordinateMap<Frame::ElementLogical, Frame::Inertial,
                                Cartoon_map_combination>
        map{{Affine{-1.0, 1.0, 0.4, 0.7}, Affine{-1.0, 1.0, -0.5, 3.2},
             Identity1D{}}};
    const auto inv_jac = map.inv_jacobian(logical_coordinates(mesh));
    const auto inertial_coords = map(logical_coordinates(mesh));

    const Variables<var_tags> volume_vars{num_pts};
    // Allocate with wrong size (one grid point too few)
    Variables<db::wrap_tags_in<Tags::deriv, var_tags, tmpl::size_t<3>,
                               Frame::Inertial>>
        d_vars_wrong{num_pts - 1};

    const size_t ghost_pts_per_face =
        (fd_order / 2 + 1) * mesh.slice_away(0).number_of_grid_points();
    DirectionMap<3, gsl::span<const double>> ghost_cell_vars{};
    DataVector ghost_dv{ghost_pts_per_face * num_vars, 0.0};
    for (const auto& dir : Direction<2>::all_directions()) {
      ghost_cell_vars[Direction<3>{dir.dimension(), dir.side()}] =
          gsl::make_span(ghost_dv.data(), ghost_dv.size());
    }

    CHECK_THROWS_WITH(
        (::fd::cartoon_partial_derivatives<var_tags>(
            make_not_null(&d_vars_wrong), volume_vars, ghost_cell_vars, mesh,
            num_vars, fd_order, inv_jac, inertial_coords)),
        Catch::Matchers::ContainsSubstring(
            "The partial derivatives Variables must have size"));
  }
#endif  // SPECTRE_DEBUG
  {
    INFO("Invalid Cartoon basis layout");
    // Build mesh with Cartoon in the first dimension (dim 0) — invalid layout
    const Mesh<3> mesh_bad_basis{
        {1, pts, 1},
        {Spectral::Basis::Cartoon, Spectral::Basis::FiniteDifference,
         Spectral::Basis::Cartoon},
        {Spectral::Quadrature::SphericalSymmetry,
         Spectral::Quadrature::CellCentered,
         Spectral::Quadrature::AxialSymmetry}};
    const size_t num_pts = mesh_bad_basis.number_of_grid_points();
    const size_t num_vars =
        Variables<var_tags>::number_of_independent_components;

    const InverseJacobian<DataVector, 3, Frame::ElementLogical, Frame::Inertial>
        inv_jac{num_pts, 0.0};
    const tnsr::I<DataVector, 3, Frame::Inertial> inertial_coords{num_pts, 0.0};

    const Variables<var_tags> volume_vars{num_pts};
    Variables<db::wrap_tags_in<Tags::deriv, var_tags, tmpl::size_t<3>,
                               Frame::Inertial>>
        d_vars{num_pts};

    const DirectionMap<3, gsl::span<const double>> ghost_cell_vars{};

    CHECK_THROWS_WITH(
        (::fd::cartoon_partial_derivatives<var_tags>(
            make_not_null(&d_vars), volume_vars, ghost_cell_vars,
            mesh_bad_basis, num_vars, fd_order, inv_jac, inertial_coords)),
        Catch::Matchers::ContainsSubstring(
            "Bases do not match valid Cartoon pattern"));
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.FiniteDifference.PartialDerivatives",
                  "[Unit][NumericalAlgorithms]") {
  test_asserts_and_errors();
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist{-1.0, 1.0};
  for (const size_t fd_order : {2_st, 4_st, 6_st, 8_st}) {
    test<1>(make_not_null(&generator), make_not_null(&dist), fd_order + 2,
            fd_order);
    test<2>(make_not_null(&generator), make_not_null(&dist), fd_order + 2,
            fd_order);
    test<3>(make_not_null(&generator), make_not_null(&dist), fd_order + 2,
            fd_order);
    if (fd_order > 2) {
      test_cartoon<true>(fd_order + 2, fd_order);
      test_cartoon<false>(fd_order + 2, fd_order);
    }
  }
}
