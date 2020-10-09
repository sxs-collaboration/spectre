// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Rotation.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Tags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Utilities/MakeArray.hpp"

namespace domain {
namespace {
template <size_t Dim>
void test_simple_tags() noexcept {
  TestHelpers::db::test_simple_tag<Tags::Domain<Dim>>("Domain");
  TestHelpers::db::test_simple_tag<Tags::InitialExtents<Dim>>("InitialExtents");
  TestHelpers::db::test_simple_tag<Tags::InitialRefinementLevels<Dim>>(
      "InitialRefinementLevels");
  TestHelpers::db::test_simple_tag<Tags::Element<Dim>>("Element");
  TestHelpers::db::test_simple_tag<Tags::Mesh<Dim>>("Mesh");
  TestHelpers::db::test_simple_tag<Tags::ElementMap<Dim>>(
      "ElementMap(Inertial)");
  TestHelpers::db::test_simple_tag<Tags::ElementMap<Dim, Frame::Grid>>(
      "ElementMap(Grid)");
  TestHelpers::db::test_simple_tag<Tags::Coordinates<Dim, Frame::Grid>>(
      "GridCoordinates");
  TestHelpers::db::test_simple_tag<Tags::Coordinates<Dim, Frame::Logical>>(
      "LogicalCoordinates");
  TestHelpers::db::test_simple_tag<Tags::Coordinates<Dim, Frame::Inertial>>(
      "InertialCoordinates");
  TestHelpers::db::test_simple_tag<
      Tags::InverseJacobian<Dim, Frame::Logical, Frame::Inertial>>(
      "InverseJacobian(Logical,Inertial)");
  TestHelpers::db::test_simple_tag<
      Tags::DetInvJacobian<Frame::Logical, Frame::Inertial>>(
      "DetInvJacobian(Logical,Inertial)");
  TestHelpers::db::test_simple_tag<Tags::InternalDirections<Dim>>(
      "InternalDirections");
  TestHelpers::db::test_simple_tag<Tags::BoundaryDirectionsInterior<Dim>>(
      "BoundaryDirectionsInterior");
  TestHelpers::db::test_simple_tag<Tags::BoundaryDirectionsExterior<Dim>>(
      "BoundaryDirectionsExterior");
  TestHelpers::db::test_simple_tag<Tags::Direction<Dim>>("Direction");
  TestHelpers::db::test_simple_tag<
      Tags::Jacobian<Dim, Frame::Logical, Frame::Inertial>>(
      "Jacobian(Logical,Inertial)");
}

template <size_t Dim>
ElementMap<Dim, Frame::Grid> element_map() noexcept;

template <>
ElementMap<1, Frame::Grid> element_map() noexcept {
  constexpr size_t dim = 1;
  const auto segment_ids = std::array<SegmentId, dim>({{SegmentId(2, 3)}});
  const CoordinateMaps::Affine first_map{-3.0, 8.7, 0.4, 5.5};
  const CoordinateMaps::Affine second_map{1.0, 8.0, -2.5, -1.0};
  const ElementId<dim> element_id(0, segment_ids);
  return ElementMap<dim, Frame::Grid>{
      element_id, make_coordinate_map_base<Frame::Logical, Frame::Grid>(
                      first_map, second_map)};
}

template <>
ElementMap<2, Frame::Grid> element_map() noexcept {
  constexpr size_t dim = 2;
  const auto segment_ids =
      std::array<SegmentId, dim>({{SegmentId(2, 3), SegmentId(1, 0)}});
  const CoordinateMaps::Rotation<dim> first_map(1.6);
  const CoordinateMaps::Rotation<dim> second_map(3.1);
  const ElementId<dim> element_id(0, segment_ids);
  return ElementMap<dim, Frame::Grid>{
      element_id, make_coordinate_map_base<Frame::Logical, Frame::Grid>(
                      first_map, second_map)};
}

template <>
ElementMap<3, Frame::Grid> element_map() noexcept {
  constexpr size_t dim = 3;
  const auto segment_ids = std::array<SegmentId, dim>(
      {{SegmentId(2, 3), SegmentId(1, 0), SegmentId(2, 1)}});
  const CoordinateMaps::Rotation<dim> first_map{M_PI_4, M_PI_4, M_PI_2};
  const CoordinateMaps::Rotation<dim> second_map{M_PI_4, M_PI_2, M_PI_2};
  const ElementId<dim> element_id(0, segment_ids);
  return ElementMap<dim, Frame::Grid>{
      element_id, make_coordinate_map_base<Frame::Logical, Frame::Grid>(
                      first_map, second_map)};
}

template <size_t Dim>
void test_compute_tags() noexcept {
  TestHelpers::db::test_compute_tag<Tags::InverseJacobianCompute<
      Tags::ElementMap<Dim>, Tags::Coordinates<Dim, Frame::Logical>>>(
      "InverseJacobian(Logical,Inertial)");
  TestHelpers::db::test_compute_tag<
      Tags::DetInvJacobianCompute<Dim, Frame::Logical, Frame::Inertial>>(
      "DetInvJacobian(Logical,Inertial)");
  TestHelpers::db::test_compute_tag<Tags::InternalDirectionsCompute<Dim>>(
      "InternalDirections");
  TestHelpers::db::test_compute_tag<
      Tags::BoundaryDirectionsInteriorCompute<Dim>>(
      "BoundaryDirectionsInterior");
  TestHelpers::db::test_compute_tag<
      Tags::BoundaryDirectionsExteriorCompute<Dim>>(
      "BoundaryDirectionsExterior");
  TestHelpers::db::test_compute_tag<Tags::MappedCoordinates<
      Tags::ElementMap<Dim>, Tags::Coordinates<Dim, Frame::Logical>>>(
      "InertialCoordinates");
  TestHelpers::db::test_compute_tag<
      Tags::JacobianCompute<Dim, Frame::Logical, Frame::Inertial>>(
      "Jacobian(Logical,Inertial)");

  auto map = element_map<Dim>();
  const tnsr::I<DataVector, Dim, Frame::Logical> logical_coords(
      make_array<Dim>(DataVector{-1.0, -0.5, 0.0, 0.5, 1.0}));
  const auto expected_inv_jacobian = map.inv_jacobian(logical_coords);
  const auto expected_jacobian = map.jacobian(logical_coords);

  const auto box = db::create<
      tmpl::list<Tags::ElementMap<Dim, Frame::Grid>,
                 Tags::Coordinates<Dim, Frame::Logical>>,
      db::AddComputeTags<
          Tags::InverseJacobianCompute<Tags::ElementMap<Dim, Frame::Grid>,
                                       Tags::Coordinates<Dim, Frame::Logical>>,
          Tags::DetInvJacobianCompute<Dim, Frame::Logical, Frame::Grid>,
          Tags::JacobianCompute<Dim, Frame::Logical, Frame::Grid>>>(
      std::move(map), logical_coords);
  CHECK_ITERABLE_APPROX(
      (db::get<Tags::InverseJacobian<Dim, Frame::Logical, Frame::Grid>>(box)),
      expected_inv_jacobian);
  CHECK_ITERABLE_APPROX(
      (db::get<Tags::DetInvJacobian<Frame::Logical, Frame::Grid>>(box)),
      determinant(expected_inv_jacobian));
  CHECK_ITERABLE_APPROX(
      (db::get<Tags::Jacobian<Dim, Frame::Logical, Frame::Grid>>(box)),
      expected_jacobian);
}

SPECTRE_TEST_CASE("Unit.Domain.Tags", "[Unit][Domain]") {
  test_simple_tags<1>();
  test_simple_tags<2>();
  test_simple_tags<3>();

  test_compute_tags<1>();
  test_compute_tags<2>();
  test_compute_tags<3>();
}
}  // namespace
}  // namespace domain
