// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Side.hpp"
#include "Domain/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/GeneralRelativity/ProjectionOperators.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"

namespace {
template <size_t SpatialDim, typename DataType>
void test_projection_operator(const DataType& used_for_size) {
  {
    tnsr::II<DataType, SpatialDim, Frame::Inertial> (*f)(
        const tnsr::II<DataType, SpatialDim, Frame::Inertial>&,
        const tnsr::I<DataType, SpatialDim, Frame::Inertial>&) =
        &gr::transverse_projection_operator<SpatialDim, Frame::Inertial,
                                            DataType>;
    pypp::check_with_random_values<1>(f, "ProjectionOperators",
                                      "transverse_projection_operator",
                                      {{{-1., 1.}}}, used_for_size);
  }

  {
    tnsr::ii<DataType, SpatialDim, Frame::Inertial> (*f)(
        const tnsr::ii<DataType, SpatialDim, Frame::Inertial>&,
        const tnsr::i<DataType, SpatialDim, Frame::Inertial>&) =
        &gr::transverse_projection_operator<SpatialDim, Frame::Inertial,
                                            DataType>;
    pypp::check_with_random_values<1>(f, "ProjectionOperators",
                                      "transverse_projection_operator",
                                      {{{-1., 1.}}}, used_for_size);
  }

  {
    tnsr::Ij<DataType, SpatialDim, Frame::Inertial> (*f)(
        const tnsr::I<DataType, SpatialDim, Frame::Inertial>&,
        const tnsr::i<DataType, SpatialDim, Frame::Inertial>&) =
        &gr::transverse_projection_operator<SpatialDim, Frame::Inertial,
                                            DataType>;
    pypp::check_with_random_values<1>(
        f, "ProjectionOperators",
        "transverse_projection_operator_mixed_from_spatial_input",
        {{{-1., 1.}}}, used_for_size);
  }

  {
    tnsr::AA<DataType, SpatialDim, Frame::Inertial> (*f)(
        const tnsr::AA<DataType, SpatialDim, Frame::Inertial>&,
        const tnsr::A<DataType, SpatialDim, Frame::Inertial>&,
        const tnsr::I<DataType, SpatialDim, Frame::Inertial>&) =
        &gr::transverse_projection_operator<SpatialDim, Frame::Inertial,
                                            DataType>;
    pypp::check_with_random_values<1>(
        f, "ProjectionOperators", "projection_operator_transverse_to_interface",
        {{{-1., 1.}}}, used_for_size);
  }

  {
    tnsr::aa<DataType, SpatialDim, Frame::Inertial> (*f)(
        const tnsr::aa<DataType, SpatialDim, Frame::Inertial>&,
        const tnsr::a<DataType, SpatialDim, Frame::Inertial>&,
        const tnsr::i<DataType, SpatialDim, Frame::Inertial>&) =
        &gr::transverse_projection_operator<SpatialDim, Frame::Inertial,
                                            DataType>;
    pypp::check_with_random_values<1>(
        f, "ProjectionOperators", "projection_operator_transverse_to_interface",
        {{{-1., 1.}}}, used_for_size);
  }

  {
    tnsr::Ab<DataType, SpatialDim, Frame::Inertial> (*f)(
        const tnsr::A<DataType, SpatialDim, Frame::Inertial>&,
        const tnsr::a<DataType, SpatialDim, Frame::Inertial>&,
        const tnsr::I<DataType, SpatialDim, Frame::Inertial>&,
        const tnsr::i<DataType, SpatialDim, Frame::Inertial>&) =
        &gr::transverse_projection_operator<SpatialDim, Frame::Inertial,
                                            DataType>;
    pypp::check_with_random_values<1>(
        f, "ProjectionOperators",
        "projection_operator_transverse_to_interface_mixed", {{{-1., 1.}}},
        used_for_size);
  }
}
}  // namespace

namespace {
using Affine = domain::CoordinateMaps::Affine;
using Affine3D = domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
using frame = Frame::Inertial;
constexpr size_t SpatialDim = 3;

// Test projection operators by comparing to values from SpEC
void test_spatial_projection_tensors_3D(
    const size_t grid_size_each_dimension,
    const std::array<double, 3>& lower_bound,
    const std::array<double, 3>& upper_bound) noexcept {
  // Setup grid
  Mesh<SpatialDim> mesh{grid_size_each_dimension, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const auto coord_map =
      domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(Affine3D{
          Affine{-1., 1., lower_bound[0], upper_bound[0]},
          Affine{-1., 1., lower_bound[1], upper_bound[1]},
          Affine{-1., 1., lower_bound[2], upper_bound[2]},
      });

  // Setup coordinates
  const auto x_logical = logical_coordinates(mesh);
  const auto x = coord_map(x_logical);
  const Direction<SpatialDim> direction(1, Side::Upper);  // +y direction
  const size_t slice_grid_points =
      mesh.extents().slice_away(direction.dimension()).product();
  const auto inertial_coords = [&slice_grid_points, &lower_bound]() {
    tnsr::I<DataVector, SpatialDim, frame> tmp(slice_grid_points, 0.);
    // +y direction
    get<1>(tmp) = 0.5;
    for (size_t i = 0; i < SpatialDim; ++i) {
      for (size_t j = 0; j < SpatialDim; ++j) {
        get<0>(tmp)[i * SpatialDim + j] =
            lower_bound[0] + 0.5 * static_cast<double>(i);
        get<2>(tmp)[i * SpatialDim + j] =
            lower_bound[2] + 0.5 * static_cast<double>(j);
      }
    }
    return tmp;
  }();

  // 1. Projection IJ
  auto local_inverse_spatial_metric =
      make_with_value<tnsr::II<DataVector, SpatialDim, Frame::Inertial>>(
          inertial_coords, 0.);
  auto local_unit_interface_normal_vector =
      make_with_value<tnsr::I<DataVector, SpatialDim, Frame::Inertial>>(
          inertial_coords, 0.);
  auto local_spatial_projection_IJ =
      make_with_value<tnsr::II<DataVector, SpatialDim, Frame::Inertial>>(
          inertial_coords, 0.);

  // Setting inverse_spatial_metric to compare with values from SpEC
  for (size_t i = 0; i < get<0>(inertial_coords).size(); ++i) {
    for (size_t j = 0; j < SpatialDim; ++j) {
      local_inverse_spatial_metric.get(0, j)[i] = 41.;
      local_inverse_spatial_metric.get(1, j)[i] = 43.;
      local_inverse_spatial_metric.get(2, j)[i] = 47.;
    }
  }
  // Setting unit_interface_normal_vector to compare with values from SpEC
  get<0>(local_unit_interface_normal_vector) = -1.;
  get<1>(local_unit_interface_normal_vector) = 1.;
  get<2>(local_unit_interface_normal_vector) = 1.;

  // Call tested function
  gr::transverse_projection_operator(
      make_not_null(&local_spatial_projection_IJ), local_inverse_spatial_metric,
      local_unit_interface_normal_vector);

  // Initialize with values from SpEC
  auto spec_spatial_projection_IJ =
      make_with_value<tnsr::II<DataVector, SpatialDim, Frame::Inertial>>(
          inertial_coords, 0.);
  {
    const std::array<double, 9> spec_vals = {
        {40., 42., 42., 42., 42., 42., 42., 42., 46.}};
    for (size_t j = 0; j < SpatialDim; ++j) {
      for (size_t k = j; k < SpatialDim; ++k) {
        spec_spatial_projection_IJ.get(j, k) =
            gsl::at(spec_vals, j * SpatialDim + k);
      }
    }
  }

  // Compare values returned to those from SpEC
  CHECK_ITERABLE_APPROX(local_spatial_projection_IJ,
                        spec_spatial_projection_IJ);

  // 2. Projection ij
  auto local_spatial_metric =
      make_with_value<tnsr::ii<DataVector, SpatialDim, Frame::Inertial>>(
          inertial_coords, 0.);
  auto local_unit_interface_normal_one_form =
      make_with_value<tnsr::i<DataVector, SpatialDim, Frame::Inertial>>(
          inertial_coords, 0.);
  auto local_spatial_projection_ij =
      make_with_value<tnsr::ii<DataVector, SpatialDim, Frame::Inertial>>(
          inertial_coords, 0.);

  // Setting inverse_spatial_metric to compare with values from SpEC
  for (size_t i = 0; i < SpatialDim; ++i) {
    local_spatial_metric.get(0, i) = 263.;
    local_spatial_metric.get(1, i) = 269.;
    local_spatial_metric.get(2, i) = 271.;
  }
  // Setting unit_interface_normal_vector to compare with values from SpEC
  get<0>(local_unit_interface_normal_one_form) = -1.;
  get<1>(local_unit_interface_normal_one_form) = 1.;
  get<2>(local_unit_interface_normal_one_form) = 1.;

  // Call tested function
  gr::transverse_projection_operator(
      make_not_null(&local_spatial_projection_ij), local_spatial_metric,
      local_unit_interface_normal_one_form);

  // Initialize with values from SpEC
  auto spec_spatial_projection_ij =
      make_with_value<tnsr::ii<DataVector, SpatialDim, Frame::Inertial>>(
          inertial_coords, 0.);
  {
    const std::array<double, 9> spec_vals = {
        {262., 264., 264., 264., 268., 268., 264., 268., 270.}};
    for (size_t j = 0; j < SpatialDim; ++j) {
      for (size_t k = j; k < SpatialDim; ++k) {
        spec_spatial_projection_ij.get(j, k) =
            gsl::at(spec_vals, j * SpatialDim + k);
      }
    }
  }

  // Compare values returned to those from SpEC
  CHECK_ITERABLE_APPROX(local_spatial_projection_ij,
                        spec_spatial_projection_ij);

  // 3. Projection Ij
  auto local_spatial_projection_Ij =
      make_with_value<tnsr::Ij<DataVector, SpatialDim, Frame::Inertial>>(
          inertial_coords, 0.);

  // Call tested function
  gr::transverse_projection_operator(
      make_not_null(&local_spatial_projection_Ij),
      local_unit_interface_normal_vector, local_unit_interface_normal_one_form);

  // Initialize with values from SpEC
  auto spec_spatial_projection_Ij =
      make_with_value<tnsr::Ij<DataVector, SpatialDim, Frame::Inertial>>(
          inertial_coords, 0.);
  {
    const std::array<double, 9> spec_vals = {
        {0., 1., 1., 1., 0., -1., 1., -1., 0.}};
    for (size_t j = 0; j < SpatialDim; ++j) {
      for (size_t k = 0; k < SpatialDim; ++k) {
        spec_spatial_projection_Ij.get(j, k) =
            gsl::at(spec_vals, j * SpatialDim + k);
      }
    }
  }

  // Compare values returned to those from SpEC
  CHECK_ITERABLE_APPROX(local_spatial_projection_Ij,
                        spec_spatial_projection_Ij);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GeneralRelativity.ProjectionOps",
                  "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "PointwiseFunctions/GeneralRelativity/");

  GENERATE_UNINITIALIZED_DOUBLE_AND_DATAVECTOR;

  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_projection_operator, (1, 2, 3));

  const size_t grid_size = 3;
  const std::array<double, 3> lower_bound{{299., -0.5, -0.5}};
  const std::array<double, 3> upper_bound{{300., 0.5, 0.5}};

  test_spatial_projection_tensors_3D(grid_size, lower_bound, upper_bound);
}
