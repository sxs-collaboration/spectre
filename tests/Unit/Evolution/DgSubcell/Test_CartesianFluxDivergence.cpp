// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Block.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DgSubcell/CartesianFluxDivergence.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace {
template <size_t Dim>
void test() {
  const size_t num_pts_1d = 5;
  const Index<Dim> subcell_extents{num_pts_1d};
  for (size_t d = 0; d < Dim; ++d) {
    CAPTURE(d);
    auto extents = make_array<Dim>(num_pts_1d);
    ++gsl::at(extents, d);
    const auto basis = make_array<Dim>(Spectral::Basis::FiniteDifference);
    auto quadrature = make_array<Dim>(Spectral::Quadrature::CellCentered);
    gsl::at(quadrature, d) = Spectral::Quadrature::FaceCentered;
    const Mesh<Dim> subcell_face_mesh{extents, basis, quadrature};

    DataVector dt_var{subcell_extents.product(), 1.2};
    const DataVector inv_jacobian{subcell_extents.product(), 5.0};
    const auto logical_coords = logical_coordinates(subcell_face_mesh);
    const double one_over_delta =
        1.0 / (get<0>(logical_coords)[1] - get<0>(logical_coords)[0]);
    const DataVector boundary_correction = 3.0 * logical_coords.get(d);
    evolution::dg::subcell::add_cartesian_flux_divergence(
        make_not_null(&dt_var), one_over_delta, inv_jacobian,
        boundary_correction, subcell_extents, d);
    const DataVector expected_dt_var{subcell_extents.product(),
                                     inv_jacobian[0] * 3.0 + 1.2};
    CHECK_ITERABLE_APPROX(dt_var, expected_dt_var);
  }
}

void test_cartoon() {
  // Helper function to create spatially varying inverse Jacobian
  const auto create_varying_inv_jacobian =
      [](const size_t num_points, const double base_value,
         const double increment) -> DataVector {
    DataVector result(num_points);
    for (size_t i = 0; i < num_points; ++i) {
      result[i] = base_value + increment * static_cast<double>(i);
    }
    return result;
  };
  // Helper function to compute expected result for a single cell
  const auto compute_cell_contribution =
      [](const double one_over_delta, const double inv_jac,
         const double weight_lower, const double flux_lower,
         const double weight_upper, const double flux_upper) -> double {
    return one_over_delta * inv_jac *
           (weight_upper * flux_upper - weight_lower * flux_lower);
  };

  const double time = 0.0;
  const std::unordered_map<
      std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};

  using Affine = domain::CoordinateMaps::Affine;
  using Affine3d =
      domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

  const Affine x_map{-1.0, 1.0, 1.0, 3.0};
  const Affine y_map{-1.0, 1.0, -1.0, 1.0};
  const Affine z_map{-1.0, 1.0, -1.0, 1.0};

  const auto block_to_inertial_coord_map =
      domain::make_coordinate_map<Frame::BlockLogical, Frame::Inertial>(
          Affine3d{x_map, y_map, z_map});
  const Block<3> block{block_to_inertial_coord_map.get_clone(), 0, {}};
  const ElementId<3> element_id{0};
  const auto logical_to_grid_map =
      ElementMap<3, Frame::Grid>{element_id, block};
  const auto grid_to_inertial_map_ptr =
      domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
          domain::CoordinateMaps::Identity<3>{});
  const auto& grid_to_inertial_map = *grid_to_inertial_map_ptr;

  {
    INFO("Spherical symmetry");
    const Index<3> subcell_extents{3, 1, 1};
    const size_t dimension = 0;

    const Mesh<3> volume_mesh{
        subcell_extents.indices(),
        {Spectral::Basis::FiniteDifference, Spectral::Basis::Cartoon,
         Spectral::Basis::Cartoon},
        {Spectral::Quadrature::CellCentered,
         Spectral::Quadrature::SphericalSymmetry,
         Spectral::Quadrature::SphericalSymmetry}};

    const auto volume_logical_coords = logical_coordinates(volume_mesh);
    const auto volume_inertial_coords = grid_to_inertial_map(
        logical_to_grid_map(volume_logical_coords), time, functions_of_time);

    Index<3> face_extents = subcell_extents;
    ++face_extents[dimension];
    const Mesh<3> face_mesh{
        face_extents.indices(),
        {Spectral::Basis::FiniteDifference, Spectral::Basis::Cartoon,
         Spectral::Basis::Cartoon},
        {Spectral::Quadrature::FaceCentered,
         Spectral::Quadrature::SphericalSymmetry,
         Spectral::Quadrature::SphericalSymmetry}};

    const auto face_logical_coords = logical_coordinates(face_mesh);
    const auto face_inertial_coords = grid_to_inertial_map(
        logical_to_grid_map(face_logical_coords), time, functions_of_time);

    const double one_over_delta = 1.0 / (get<0>(volume_logical_coords)[1] -
                                         get<0>(volume_logical_coords)[0]);

    // Test with quadratic flux F^x = x^2
    DataVector boundary_correction(face_inertial_coords.get(0).size());
    for (size_t i = 0; i < boundary_correction.size(); ++i) {
      boundary_correction[i] = square(get<0>(face_inertial_coords)[i]);
    }

    DataVector dt_var(subcell_extents.product(), 0.0);
    const DataVector inv_jacobian =
        create_varying_inv_jacobian(subcell_extents.product(), 0.5, 0.2);

    evolution::dg::subcell::add_cartoon_cartesian_flux_divergence(
        make_not_null(&dt_var), one_over_delta, inv_jacobian,
        boundary_correction, subcell_extents, dimension, volume_inertial_coords,
        logical_to_grid_map, grid_to_inertial_map, time, functions_of_time);

    // Compute expected results for spherical symmetry with F^x = x^2
    DataVector expected_dt_var(3);

    // Cell data: [x_vol, x_face_lower, x_face_upper]
    const std::array<std::array<double, 3>, 3> cell_data = {{
        {4.0 / 3.0, 1.0, 5.0 / 3.0},  // Cell 0
        {2.0, 5.0 / 3.0, 7.0 / 3.0},  // Cell 1
        {8.0 / 3.0, 7.0 / 3.0, 3.0}   // Cell 2
    }};

    for (size_t i = 0; i < 3; ++i) {
      const double x_vol = gsl::at(cell_data, i)[0];
      const double x_face_lower = gsl::at(cell_data, i)[1];
      const double x_face_upper = gsl::at(cell_data, i)[2];

      const double f_lower = square(x_face_lower);
      const double f_upper = square(x_face_upper);
      const double weight_lower = square(x_face_lower) / square(x_vol);
      const double weight_upper = square(x_face_upper) / square(x_vol);

      expected_dt_var[i] = compute_cell_contribution(
          one_over_delta, inv_jacobian[i], weight_lower, f_lower, weight_upper,
          f_upper);
    }

    CHECK_ITERABLE_APPROX(dt_var, expected_dt_var);
  }
  {
    INFO("Axial symmetry - dimension 0");
    const Index<3> subcell_extents{2, 2, 1};
    const size_t dimension = 0;

    const Mesh<3> volume_mesh{
        subcell_extents.indices(),
        {Spectral::Basis::FiniteDifference, Spectral::Basis::FiniteDifference,
         Spectral::Basis::Cartoon},
        {Spectral::Quadrature::CellCentered, Spectral::Quadrature::CellCentered,
         Spectral::Quadrature::AxialSymmetry}};

    const auto volume_logical_coords = logical_coordinates(volume_mesh);
    const auto volume_inertial_coords = grid_to_inertial_map(
        logical_to_grid_map(volume_logical_coords), time, functions_of_time);

    // Face mesh
    Index<3> face_extents = subcell_extents;
    ++face_extents[dimension];
    const Mesh<3> face_mesh{
        face_extents.indices(),
        {Spectral::Basis::FiniteDifference, Spectral::Basis::FiniteDifference,
         Spectral::Basis::Cartoon},
        {Spectral::Quadrature::FaceCentered, Spectral::Quadrature::CellCentered,
         Spectral::Quadrature::AxialSymmetry}};

    const auto face_logical_coords = logical_coordinates(face_mesh);
    const auto face_inertial_coords = grid_to_inertial_map(
        logical_to_grid_map(face_logical_coords), time, functions_of_time);

    const double one_over_delta = 1.0 / (get<0>(volume_logical_coords)[1] -
                                         get<0>(volume_logical_coords)[0]);

    // Test with flux F^x = x
    const DataVector boundary_correction = get<0>(face_inertial_coords);

    DataVector dt_var(subcell_extents.product(), 0.0);
    const DataVector inv_jacobian =
        create_varying_inv_jacobian(subcell_extents.product(), 0.8, 0.1);

    evolution::dg::subcell::add_cartoon_cartesian_flux_divergence(
        make_not_null(&dt_var), one_over_delta, inv_jacobian,
        boundary_correction, subcell_extents, dimension, volume_inertial_coords,
        logical_to_grid_map, grid_to_inertial_map, time, functions_of_time);

    DataVector expected_dt_var(4);

    // Cell (0,0): x_vol = 1.5, x_face = [1, 2], inv_jac = 0.8
    const double x_vol_00 = 1.5;
    const double weight_lower_00 = 1.0 / x_vol_00;
    const double weight_upper_00 = 2.0 / x_vol_00;
    expected_dt_var[0] = one_over_delta * inv_jacobian[0] *
                         (weight_upper_00 * 2.0 - weight_lower_00 * 1.0);

    // Cell (1,0): x_vol = 2.5, x_face = [2, 3], inv_jac = 0.9
    const double x_vol_10 = 2.5;
    const double weight_lower_10 = 2.0 / x_vol_10;
    const double weight_upper_10 = 3.0 / x_vol_10;
    expected_dt_var[1] = one_over_delta * inv_jacobian[1] *
                         (weight_upper_10 * 3.0 - weight_lower_10 * 2.0);

    // Cells (0,1) and (1,1): same x-coords but different inv_jacobian
    expected_dt_var[2] = one_over_delta * inv_jacobian[2] *
                         (weight_upper_00 * 2.0 - weight_lower_00 * 1.0);
    expected_dt_var[3] = one_over_delta * inv_jacobian[3] *
                         (weight_upper_10 * 3.0 - weight_lower_10 * 2.0);

    CHECK_ITERABLE_APPROX(dt_var, expected_dt_var);
  }
  {
    // Should behave like regular Cartesian
    INFO("Axial symmetry - dimension 1");
    const Index<3> subcell_extents{2, 2, 1};
    const size_t dimension = 1;

    const Mesh<3> volume_mesh{
        subcell_extents.indices(),
        {Spectral::Basis::FiniteDifference, Spectral::Basis::FiniteDifference,
         Spectral::Basis::Cartoon},
        {Spectral::Quadrature::CellCentered, Spectral::Quadrature::CellCentered,
         Spectral::Quadrature::AxialSymmetry}};

    const auto volume_logical_coords = logical_coordinates(volume_mesh);
    const auto volume_inertial_coords = grid_to_inertial_map(
        logical_to_grid_map(volume_logical_coords), time, functions_of_time);

    Index<3> face_extents = subcell_extents;
    ++face_extents[dimension];
    const Mesh<3> face_mesh{
        face_extents.indices(),
        {Spectral::Basis::FiniteDifference, Spectral::Basis::FiniteDifference,
         Spectral::Basis::Cartoon},
        {Spectral::Quadrature::CellCentered, Spectral::Quadrature::FaceCentered,
         Spectral::Quadrature::AxialSymmetry}};

    const auto face_logical_coords = logical_coordinates(face_mesh);
    const auto face_inertial_coords = grid_to_inertial_map(
        logical_to_grid_map(face_logical_coords), time, functions_of_time);

    const double one_over_delta = 1.0 / (get<1>(volume_logical_coords)[2] -
                                         get<1>(volume_logical_coords)[0]);

    // Test with flux F^y = y² + x
    DataVector boundary_correction(face_inertial_coords.get(1).size());
    for (size_t i = 0; i < boundary_correction.size(); ++i) {
      const double y_coord = get<1>(face_inertial_coords)[i];
      const double x_coord = get<0>(face_inertial_coords)[i];
      boundary_correction[i] = square(y_coord) + x_coord;
    }

    DataVector dt_var(subcell_extents.product(), 0.0);
    const DataVector inv_jacobian =
        create_varying_inv_jacobian(subcell_extents.product(), 0.6, 0.15);

    evolution::dg::subcell::add_cartoon_cartesian_flux_divergence(
        make_not_null(&dt_var), one_over_delta, inv_jacobian,
        boundary_correction, subcell_extents, dimension, volume_inertial_coords,
        logical_to_grid_map, grid_to_inertial_map, time, functions_of_time);

    // For axial symmetry in y-direction: no coordinate weighting (standard
    // finite difference)
    DataVector expected_dt_var(4);

    for (size_t vol_idx = 0; vol_idx < 4; ++vol_idx) {
      const size_t i = vol_idx % 2;                   // x-index
      const size_t j = vol_idx / 2;                   // y-index
      const size_t face_lower_idx = i + j * 2;        // Lower y-face
      const size_t face_upper_idx = i + (j + 1) * 2;  // Upper y-face

      // No coordinate weighting in y-direction (weight = 1.0)
      expected_dt_var[vol_idx] =
          compute_cell_contribution(one_over_delta, inv_jacobian[vol_idx], 1.0,
                                    boundary_correction[face_lower_idx], 1.0,
                                    boundary_correction[face_upper_idx]);
    }
    CHECK_ITERABLE_APPROX(dt_var, expected_dt_var);
  }
}

void test_validation_errors() {
#ifdef SPECTRE_DEBUG
  {
    INFO("Test 1D validation errors");
    const Index<1> subcell_extents{3};
    {
      INFO("Test wrong dt_var size");
      DataVector dt_var_wrong_size(2, 0.0);  // Should be 3
      const DataVector inv_jacobian(3, 1.0);
      const DataVector boundary_correction(4, 1.0);
      CHECK_THROWS_WITH(
          evolution::dg::subcell::add_cartesian_flux_divergence(
              make_not_null(&dt_var_wrong_size), 1.0, inv_jacobian,
              boundary_correction, subcell_extents, 0),
          Catch::Matchers::ContainsSubstring(
              "dt_var size 2 does not match expected volume points 3"));
    }
    {
      INFO("Test wrong inv_jacobian size");
      DataVector dt_var(3, 0.0);
      const DataVector inv_jacobian_wrong_size(2, 1.0);  // Should be 3
      const DataVector boundary_correction(4, 1.0);
      CHECK_THROWS_WITH(
          evolution::dg::subcell::add_cartesian_flux_divergence(
              make_not_null(&dt_var), 1.0, inv_jacobian_wrong_size,
              boundary_correction, subcell_extents, 0),
          Catch::Matchers::ContainsSubstring(
              "inv_jacobian size 2 does not match expected volume points 3"));
    }
    {
      INFO("Test wrong boundary_correction size");
      DataVector dt_var(3, 0.0);
      const DataVector inv_jacobian(3, 1.0);
      const DataVector boundary_correction_wrong_size(3, 1.0);
      CHECK_THROWS_WITH(evolution::dg::subcell::add_cartesian_flux_divergence(
                            make_not_null(&dt_var), 1.0, inv_jacobian,
                            boundary_correction_wrong_size, subcell_extents, 0),
                        Catch::Matchers::ContainsSubstring(
                            "boundary_correction size 3 does not match "
                            "expected face points 4"));
    }
    {
      INFO("Test invalid dimension");
      DataVector dt_var(3, 0.0);
      const DataVector inv_jacobian(3, 1.0);
      const DataVector boundary_correction(4, 1.0);
      CHECK_THROWS_WITH(
          evolution::dg::subcell::add_cartesian_flux_divergence(
              make_not_null(&dt_var), 1.0, inv_jacobian, boundary_correction,
              subcell_extents, 1),
          Catch::Matchers::ContainsSubstring("dimension must be 0 but is 1"));
    }
  }
  {
    INFO("Test 2D validation errors");
    const Index<2> subcell_extents{3, 2};  // 6 volume points
    {
      INFO("Test wrong boundary_correction size for dimension 0");
      DataVector dt_var(6, 0.0);
      const DataVector inv_jacobian(6, 1.0);
      const DataVector boundary_correction_wrong_size(
          7, 1.0);  // Should be 8 (4x2)
      CHECK_THROWS_WITH(evolution::dg::subcell::add_cartesian_flux_divergence(
                            make_not_null(&dt_var), 1.0, inv_jacobian,
                            boundary_correction_wrong_size, subcell_extents, 0),
                        Catch::Matchers::ContainsSubstring(
                            "boundary_correction size 7 does not match "
                            "expected face points 8"));
    }
    {
      INFO("Test invalid dimension");
      DataVector dt_var(6, 0.0);
      const DataVector inv_jacobian(6, 1.0);
      const DataVector boundary_correction(8, 1.0);
      CHECK_THROWS_WITH(evolution::dg::subcell::add_cartesian_flux_divergence(
                            make_not_null(&dt_var), 1.0, inv_jacobian,
                            boundary_correction, subcell_extents, 2),
                        Catch::Matchers::ContainsSubstring(
                            "dimension must be 0 or 1 but is 2"));
    }
  }
  {
    INFO("Test 3D validation errors");
    const Index<3> subcell_extents{2, 2, 2};  // 8 volume points
    {
      INFO("Test invalid dimension");
      DataVector dt_var(8, 0.0);
      const DataVector inv_jacobian(8, 1.0);
      const DataVector boundary_correction(12, 1.0);  // 3x2x2 for dim 0
      CHECK_THROWS_WITH(evolution::dg::subcell::add_cartesian_flux_divergence(
                            make_not_null(&dt_var), 1.0, inv_jacobian,
                            boundary_correction, subcell_extents, 3),
                        Catch::Matchers::ContainsSubstring(
                            "dimension must be 0, 1, or 2 but is 3"));
    }
  }
#endif  // SPECTRE_DEBUG
  const double time = 0.0;
  const std::unordered_map<
      std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};

  using Affine = domain::CoordinateMaps::Affine;
  using Identity = domain::CoordinateMaps::Identity<1>;
  using Affine3d =
      domain::CoordinateMaps::ProductOf3Maps<Affine, Identity, Identity>;

  const Affine x_map{-1.0, 1.0, 1.0, 3.0};

  const auto block_to_inertial_coord_map =
      domain::make_coordinate_map<Frame::BlockLogical, Frame::Inertial>(
          Affine3d{x_map, Identity{}, Identity{}});
  const Block<3> block{block_to_inertial_coord_map.get_clone(), 0, {}};
  const ElementId<3> element_id{0};
  const auto logical_to_grid_map =
      ElementMap<3, Frame::Grid>{element_id, block};
  const auto grid_to_inertial_map_ptr =
      domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
          domain::CoordinateMaps::Identity<3>{});
  const auto& grid_to_inertial_map = *grid_to_inertial_map_ptr;
  {
    INFO("Test cartoon - contains zero coordinate");
    const Index<3> subcell_extents{3, 1, 1};
    const size_t dimension = 0;

    // Create coordinates that contain x=0
    tnsr::I<DataVector, 3, Frame::Inertial> coords_with_zero(
        subcell_extents.product());
    get<0>(coords_with_zero) = DataVector{0.0, 1.0, 2.0};  // Contains x=0
    get<1>(coords_with_zero) = DataVector{0.0, 0.0, 0.0};
    get<2>(coords_with_zero) = DataVector{0.0, 0.0, 0.0};

    DataVector dt_var(subcell_extents.product(), 0.0);
    const DataVector inv_jacobian(subcell_extents.product(), 1.0);
    const DataVector boundary_correction(4, 1.0);

    CHECK_THROWS_WITH(
        evolution::dg::subcell::add_cartoon_cartesian_flux_divergence(
            make_not_null(&dt_var), 1.0, inv_jacobian, boundary_correction,
            subcell_extents, dimension, coords_with_zero, logical_to_grid_map,
            grid_to_inertial_map, time, functions_of_time),
        Catch::Matchers::ContainsSubstring("Element contains x=0 for subcell"));
  }
#ifdef SPECTRE_DEBUG
  {
    INFO("Test cartoon - wrong z extent");
    const Index<3> wrong_subcell_extents{3, 1, 2};  // z should be 1
    const size_t dimension = 0;

    const Mesh<3> volume_mesh{
        wrong_subcell_extents.indices(),
        {Spectral::Basis::FiniteDifference, Spectral::Basis::Cartoon,
         Spectral::Basis::FiniteDifference},
        {Spectral::Quadrature::CellCentered,
         Spectral::Quadrature::SphericalSymmetry,
         Spectral::Quadrature::CellCentered}};

    const auto volume_logical_coords = logical_coordinates(volume_mesh);
    const auto volume_inertial_coords = grid_to_inertial_map(
        logical_to_grid_map(volume_logical_coords), time, functions_of_time);

    DataVector dt_var(wrong_subcell_extents.product(), 0.0);
    const DataVector inv_jacobian(wrong_subcell_extents.product(), 1.0);
    const DataVector boundary_correction(8, 1.0);  // 4x1x2

    CHECK_THROWS_WITH(
        evolution::dg::subcell::add_cartoon_cartesian_flux_divergence(
            make_not_null(&dt_var), 1.0, inv_jacobian, boundary_correction,
            wrong_subcell_extents, dimension, volume_inertial_coords,
            logical_to_grid_map, grid_to_inertial_map, time, functions_of_time),
        Catch::Matchers::ContainsSubstring(
            "Expecting extent = 1 in third dimension"));
  }
  {
    INFO("Test cartoon - wrong x extent");
    const Index<3> wrong_subcell_extents{1, 2, 1};  // x should not be 1
    const size_t dimension = 0;

    const Mesh<3> volume_mesh{
        wrong_subcell_extents.indices(),
        {Spectral::Basis::FiniteDifference, Spectral::Basis::FiniteDifference,
         Spectral::Basis::Cartoon},
        {Spectral::Quadrature::CellCentered, Spectral::Quadrature::CellCentered,
         Spectral::Quadrature::AxialSymmetry}};

    const auto volume_logical_coords = logical_coordinates(volume_mesh);
    const auto volume_inertial_coords = grid_to_inertial_map(
        logical_to_grid_map(volume_logical_coords), time, functions_of_time);

    DataVector dt_var(wrong_subcell_extents.product(), 0.0);
    const DataVector inv_jacobian(wrong_subcell_extents.product(), 1.0);
    const DataVector boundary_correction(4, 1.0);  // 2x2x1

    CHECK_THROWS_WITH(
        evolution::dg::subcell::add_cartoon_cartesian_flux_divergence(
            make_not_null(&dt_var), 1.0, inv_jacobian, boundary_correction,
            wrong_subcell_extents, dimension, volume_inertial_coords,
            logical_to_grid_map, grid_to_inertial_map, time, functions_of_time),
        Catch::Matchers::ContainsSubstring(
            "Expecting extent != 1 in first dimension"));
  }
  {
    INFO("Test cartoon - invalid dimension for spherical");
    const Index<3> subcell_extents{3, 1, 1};  // Spherical symmetry
    const size_t invalid_dimension = 1;       // Should be 0 for spherical

    const Mesh<3> volume_mesh{
        subcell_extents.indices(),
        {Spectral::Basis::FiniteDifference, Spectral::Basis::Cartoon,
         Spectral::Basis::Cartoon},
        {Spectral::Quadrature::CellCentered,
         Spectral::Quadrature::SphericalSymmetry,
         Spectral::Quadrature::SphericalSymmetry}};

    const auto volume_logical_coords = logical_coordinates(volume_mesh);
    const auto volume_inertial_coords = grid_to_inertial_map(
        logical_to_grid_map(volume_logical_coords), time, functions_of_time);

    DataVector dt_var(subcell_extents.product(), 0.0);
    const DataVector inv_jacobian(subcell_extents.product(), 1.0);
    const DataVector boundary_correction(
        6,
        1.0);  // Should actually be different size but testing dimension first

    CHECK_THROWS_WITH(
        evolution::dg::subcell::add_cartoon_cartesian_flux_divergence(
            make_not_null(&dt_var), 1.0, inv_jacobian, boundary_correction,
            subcell_extents, invalid_dimension, volume_inertial_coords,
            logical_to_grid_map, grid_to_inertial_map, time, functions_of_time),
        Catch::Matchers::ContainsSubstring(
            "Using cartoon derivatives with spherical symmetry, expecting "
            "dimension = 0"));
  }
  {
    INFO("Test cartoon - invalid dimension for axial");
    const Index<3> subcell_extents{3, 2, 1};  // Axial symmetry
    const size_t invalid_dimension = 2;       // Should be 0 or 1 for axial

    const Mesh<3> volume_mesh{
        subcell_extents.indices(),
        {Spectral::Basis::FiniteDifference, Spectral::Basis::FiniteDifference,
         Spectral::Basis::Cartoon},
        {Spectral::Quadrature::CellCentered, Spectral::Quadrature::CellCentered,
         Spectral::Quadrature::AxialSymmetry}};

    const auto volume_logical_coords = logical_coordinates(volume_mesh);
    const auto volume_inertial_coords = grid_to_inertial_map(
        logical_to_grid_map(volume_logical_coords), time, functions_of_time);

    DataVector dt_var(subcell_extents.product(), 0.0);
    const DataVector inv_jacobian(subcell_extents.product(), 1.0);
    const DataVector boundary_correction(12, 1.0);

    CHECK_THROWS_WITH(
        evolution::dg::subcell::add_cartoon_cartesian_flux_divergence(
            make_not_null(&dt_var), 1.0, inv_jacobian, boundary_correction,
            subcell_extents, invalid_dimension, volume_inertial_coords,
            logical_to_grid_map, grid_to_inertial_map, time, functions_of_time),
        Catch::Matchers::ContainsSubstring(
            "Using cartoon derivatives with axial symmetry, expecting "
            "dimension = 0 or 1"));
  }
#endif  // SPECTRE_DEBUG
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.FD.CartesianFluxDivergence",
                  "[Evolution][Unit]") {
  test<1>();
  test<2>();
  test<3>();
  test_cartoon();
  test_validation_errors();
}
