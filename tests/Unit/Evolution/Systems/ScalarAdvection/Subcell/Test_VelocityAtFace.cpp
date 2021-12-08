// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
#include "Evolution/DgSubcell/Tags/OnSubcellFaces.hpp"
#include "Evolution/Initialization/Tags.hpp"
#include "Evolution/Systems/ScalarAdvection/Subcell/VelocityAtFace.hpp"
#include "Evolution/Systems/ScalarAdvection/Tags.hpp"
#include "Evolution/Systems/ScalarAdvection/VelocityField.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/CloneUniquePtrs.hpp"

namespace {

template <size_t Dim>
void test() {
  // type aliases for the face tensor
  using velocity_field = ::ScalarAdvection::Tags::VelocityField<Dim>;
  using subcell_velocity_field =
      ::evolution::dg::subcell::Tags::Inactive<velocity_field>;
  using subcell_faces_velocity_field =
      ::evolution::dg::subcell::Tags::OnSubcellFaces<velocity_field, Dim>;

  using vars = typename velocity_field::type;
  using subcell_vars = typename subcell_velocity_field::type;
  using face_vars = typename subcell_faces_velocity_field::type::value_type;

  const size_t num_pts = 5;
  const Mesh<Dim> subcell_mesh{num_pts, Spectral::Basis::FiniteDifference,
                               Spectral::Quadrature::CellCentered};

  const double time = 0.0;
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};

  // create a DataBox
  auto box = db::create<db::AddSimpleTags<
      Initialization::Tags::InitialTime,
      domain::Tags::FunctionsOfTimeInitialize,
      domain::Tags::ElementMap<Dim, Frame::Grid>,
      domain::CoordinateMaps::Tags::CoordinateMap<Dim, Frame::Grid,
                                                  Frame::Inertial>,
      evolution::dg::subcell::Tags::Mesh<Dim>,
      evolution::dg::subcell::Tags::Coordinates<Dim, Frame::ElementLogical>,
      subcell_velocity_field, subcell_faces_velocity_field>>(
      time, clone_unique_ptrs(functions_of_time),
      ElementMap<Dim, Frame::Grid>{
          ElementId<Dim>{0},
          domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(
              domain::CoordinateMaps::Identity<Dim>{})},
      domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
          domain::CoordinateMaps::Identity<Dim>{}),
      subcell_mesh, logical_coordinates(subcell_mesh), subcell_vars{},
      std::array<face_vars, Dim>{});

  // apply the mutator
  db::mutate_apply<ScalarAdvection::subcell::VelocityAtFace<Dim>>(
      make_not_null(&box));

  // construct cell-centered inertial coordinates for testing
  const ElementMap<Dim, Frame::Grid> logical_to_grid_map{
      ElementId<Dim>{0},
      domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(
          domain::CoordinateMaps::Identity<Dim>{})};
  const auto grid_to_inertial_map =
      domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
          domain::CoordinateMaps::Identity<Dim>{});
  const auto cell_centered_inertial_coords = (*grid_to_inertial_map)(
      logical_to_grid_map(logical_coordinates(subcell_mesh)), time,
      functions_of_time);

  // compute the velocity field at the cell-centered grid points
  vars expected_no_prefix_cell_centered_vars{
      subcell_mesh.number_of_grid_points()};
  ::ScalarAdvection::Tags::VelocityFieldCompute<Dim>::function(
      make_not_null(&expected_no_prefix_cell_centered_vars),
      cell_centered_inertial_coords);
  const subcell_vars expected_cell_centered_vars{
      std::move(expected_no_prefix_cell_centered_vars)};

  // check cell-centered values
  CHECK_ITERABLE_APPROX(db::get<subcell_velocity_field>(box),
                        expected_cell_centered_vars);

  // check face-centered values
  for (size_t i = 0; i < Dim; ++i) {
    // construct face-centered inertial coordinates
    const auto basis = subcell_mesh.basis();
    auto quadrature = subcell_mesh.quadrature();
    // The following method of constructing the extents std::array avoids a
    // suspected compiler bug when building on Apple Silicon. The apparent bug
    // causes the Mesh<Dim> constructor to ignore modifications to the
    // extents array's components, so here the array is constructed in a way
    // that never modifies a component once set.
    std::array<size_t, Dim> extents{};
    for (size_t j = 0; j < Dim; ++j) {
      if (j == i) {
        gsl::at(extents, j) = subcell_mesh.extents(0) + 1;
      } else {
        gsl::at(extents, j) = subcell_mesh.extents(0);
      }
    }
    gsl::at(quadrature, i) = Spectral::Quadrature::FaceCentered;
    const Mesh<Dim> face_centered_mesh{extents, basis, quadrature};
    const auto face_logical_coords = logical_coordinates(face_centered_mesh);
    const auto face_inertial_coords = (*grid_to_inertial_map)(
        logical_to_grid_map(face_logical_coords), time, functions_of_time);

    // compute the velocity field at the face-centered grid points
    face_vars expected_face_vars{face_centered_mesh.number_of_grid_points()};
    ::ScalarAdvection::Tags::VelocityFieldCompute<Dim>::function(
        make_not_null(&expected_face_vars), face_inertial_coords);

    // check
    CHECK_ITERABLE_APPROX(
        gsl::at(db::get<subcell_faces_velocity_field>(box), i),
        expected_face_vars);
  }
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.ScalarAdvection.Subcell.VelocityAtFace",
    "[Unit][Evolution]") {
  test<1>();
  test<2>();
}
