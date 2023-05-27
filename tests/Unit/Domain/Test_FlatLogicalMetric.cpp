// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SliceIterator.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Domain/FlatLogicalMetric.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {

template <size_t Dim>
auto make_affine_coord_map() {
  using AffineMap = domain::CoordinateMaps::Affine;
  if constexpr (Dim == 1) {
    return domain::CoordinateMap<Frame::ElementLogical, Frame::Inertial,
                                 AffineMap>{{-1., 1., 0., M_PI}};
  } else if constexpr (Dim == 2) {
    using AffineMap2D =
        domain::CoordinateMaps::ProductOf2Maps<AffineMap, AffineMap>;
    return domain::CoordinateMap<Frame::ElementLogical, Frame::Inertial,
                                 AffineMap2D>{
        {{-1., 1., 0., M_PI}, {-1., 1., 0., M_PI_2}}};
  } else {
    using AffineMap3D =
        domain::CoordinateMaps::ProductOf3Maps<AffineMap, AffineMap, AffineMap>;
    return domain::CoordinateMap<Frame::ElementLogical, Frame::Inertial,
                                 AffineMap3D>{
        {{-1., 1., 0., M_PI}, {-1., 1., 0., M_PI_2}, {-1., 1., 0., M_PI_4}}};
  }
}

template <size_t Dim>
void test_flat_logical_metric() {
  CAPTURE(Dim);
  const Mesh<Dim> mesh{3, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
  const size_t num_points = mesh.number_of_grid_points();
  const auto logical_coords = logical_coordinates(mesh);
  const auto coord_map = make_affine_coord_map<Dim>();
  const auto inertial_coords = coord_map(logical_coords);
  const auto jacobian = coord_map.jacobian(logical_coords);
  const auto inv_jacobian = coord_map.inv_jacobian(logical_coords);
  tnsr::ii<DataVector, Dim, Frame::ElementLogical> flat_logical_metric{
      num_points};
  domain::flat_logical_metric(make_not_null(&flat_logical_metric), jacobian);
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j < Dim; ++j) {
      CHECK_ITERABLE_APPROX(
          flat_logical_metric.get(i, j),
          DataVector(num_points, i == j ? square(M_PI_2 / pow(2, i)) : 0.));
    }
  }
  {
    INFO("Test the compute tag");
    TestHelpers::db::test_compute_tag<
        domain::Tags::FlatLogicalMetricCompute<Dim>>("FlatLogicalMetric");
    const auto box = db::create<
        db::AddSimpleTags<domain::Tags::InverseJacobian<
            Dim, Frame::ElementLogical, Frame::Inertial>>,
        db::AddComputeTags<domain::Tags::FlatLogicalMetricCompute<Dim>>>(
        inv_jacobian);
    CHECK_ITERABLE_APPROX(get<domain::Tags::FlatLogicalMetric<Dim>>(box),
                          flat_logical_metric);
  }
}

void test_equiangular() {
  INFO("Equiangular");
  // Construct a 2D equiangular Wedge map
  const domain::CoordinateMap<Frame::ElementLogical, Frame::Inertial,
                              domain::CoordinateMaps::Wedge<2>>
      coord_map{{1., 2., 1., 1., {}, true}};
  using CoordAxis = domain::CoordinateMaps::detail::WedgeCoordOrientation<2>;
  // Set up a grid
  const Mesh<2> mesh{6, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  const size_t num_points = mesh.number_of_grid_points();
  const auto logical_coords = logical_coordinates(mesh);
  const auto jacobian = coord_map.jacobian(logical_coords);
  tnsr::ii<DataVector, 2, Frame::ElementLogical> flat_logical_metric{
      num_points};
  domain::flat_logical_metric(make_not_null(&flat_logical_metric), jacobian);
  // Check that the flat logical metric is constant along the angular direction
  const DataVector& angular_data =
      get<CoordAxis::polar_coord, CoordAxis::polar_coord>(flat_logical_metric);
  CAPTURE(angular_data);
  for (size_t slice = 0; slice < mesh.extents(CoordAxis::radial_coord);
       ++slice) {
    double constant_value = std::numeric_limits<double>::signaling_NaN();
    bool first = true;
    for (SliceIterator si(mesh.extents(), CoordAxis::radial_coord, slice); si;
         ++si) {
      if (first) {
        constant_value = angular_data[si.volume_offset()];
        first = false;
      } else {
        CHECK(angular_data[si.volume_offset()] == approx(constant_value));
      }
    }
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.FlatLogicalMetric", "[Unit][Domain]") {
  test_flat_logical_metric<1>();
  test_flat_logical_metric<2>();
  test_flat_logical_metric<3>();
  test_equiangular();
}
