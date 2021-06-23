// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ProductMaps.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ProductMaps.tpp"
#include "Domain/CoordinateMaps/TimeDependent/Translation.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/OnSubcellFaces.hpp"
#include "Evolution/Initialization/InitialData.hpp"
#include "Evolution/Initialization/Tags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/GrTagsForHydro.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "Framework/TestHelpers.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Tov.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/TovStar.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Utilities/CloneUniquePtrs.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.ValenciaDivClean.Subcell.GrTagsForHydro",
    "[Unit][Evolution]") {
  using System = grmhd::ValenciaDivClean::System;
  using gr_tag = typename System::spacetime_variables_tag;
  using subcell_gr_tag = evolution::dg::subcell::Tags::Inactive<gr_tag>;
  using subcell_faces_gr_tag = evolution::dg::subcell::Tags::OnSubcellFaces<
      typename System::flux_spacetime_variables_tag, 3>;
  using GrVars = typename gr_tag::type;
  using SubcellGrVars = typename subcell_gr_tag::type;
  using FaceGrVars = typename subcell_faces_gr_tag::type::value_type;

  using Translation = domain::CoordinateMaps::TimeDependent::Translation;
  using Identity2D = domain::CoordinateMaps::Identity<2>;
  using Solution =
      RelativisticEuler::Solutions::TovStar<gr::Solutions::TovSolution>;
  const Mesh<3> subcell_mesh{5, Spectral::Basis::FiniteDifference,
                             Spectral::Quadrature::CellCentered};
  const double time = 0.1;
  const auto logical_coords = logical_coordinates(subcell_mesh);
  constexpr double central_density = 1.28e-3;
  constexpr double polytropic_constant = 100.0;
  constexpr double polytropic_exponent = 2.0;
  const Solution solution{central_density, polytropic_constant,
                          polytropic_exponent};

  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};
  functions_of_time["Translation"] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
          0.0, std::array<DataVector, 3>{{{0.0}, {2.3}, {0.0}}}, 10.0);

  auto box = db::create<db::AddSimpleTags<
      Initialization::Tags::InitialTime, evolution::dg::subcell::Tags::Mesh<3>,
      domain::Tags::ElementMap<3, Frame::Grid>,
      domain::CoordinateMaps::Tags::CoordinateMap<3, Frame::Grid,
                                                  Frame::Inertial>,
      domain::Tags::FunctionsOfTime,
      evolution::dg::subcell::Tags::Coordinates<3, Frame::Logical>,
      Tags::AnalyticSolution<Solution>, subcell_gr_tag, subcell_faces_gr_tag>>(
      time, subcell_mesh,
      ElementMap<3, Frame::Grid>{
          ElementId<3>{0},
          domain::make_coordinate_map_base<Frame::Logical, Frame::Grid>(
              domain::CoordinateMaps::Identity<3>{})},
      domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
          domain::CoordinateMaps::TimeDependent::ProductOf2Maps<Translation,
                                                                Identity2D>(
              Translation{"Translation"}, Identity2D{})),
      clone_unique_ptrs(functions_of_time), logical_coords,
      Solution{central_density, polytropic_constant, polytropic_exponent},
      SubcellGrVars{}, std::array<FaceGrVars, 3>{});

  db::mutate_apply<Initialization::subcell::GrTagsForHydro<System, 3>>(
      make_not_null(&box));

  const ElementMap<3, Frame::Grid> logical_to_grid_map{
      ElementId<3>{0},
      domain::make_coordinate_map_base<Frame::Logical, Frame::Grid>(
          domain::CoordinateMaps::Identity<3>{})};
  const auto grid_to_inertial_map =
      domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
          domain::CoordinateMaps::TimeDependent::ProductOf2Maps<Translation,
                                                                Identity2D>(
              Translation{"Translation"}, Identity2D{}));

  // Check cell-centered values
  const auto cell_centered_inertial_coords = (*grid_to_inertial_map)(
      logical_to_grid_map(logical_coords), time, functions_of_time);

  GrVars expected_no_prefix_cell_centered_gr_vars{
      subcell_mesh.number_of_grid_points()};
  expected_no_prefix_cell_centered_gr_vars.assign_subset(solution.variables(
      cell_centered_inertial_coords, time, GrVars::tags_list{}));
  const SubcellGrVars expected_centered_gr_vars{
      std::move(expected_no_prefix_cell_centered_gr_vars)};

  CHECK_VARIABLES_APPROX(db::get<subcell_gr_tag>(box),
                         expected_centered_gr_vars);

  // Check face-centered values
  for (size_t i = 0; i < 3; ++i) {
    const auto basis = subcell_mesh.basis();
    auto quadrature = subcell_mesh.quadrature();
    auto extents = make_array<3>(subcell_mesh.extents(0));
    gsl::at(extents, i) += 1;
    gsl::at(quadrature, i) = Spectral::Quadrature::FaceCentered;
    const Mesh<3> face_centered_mesh{extents, basis, quadrature};
    const auto face_logical_coords = logical_coordinates(face_centered_mesh);
    const auto face_inertial_coords = (*grid_to_inertial_map)(
        logical_to_grid_map(face_logical_coords), time, functions_of_time);
    FaceGrVars expected_face_vars{face_centered_mesh.number_of_grid_points()};
    expected_face_vars.assign_subset(solution.variables(
        face_inertial_coords, time, typename FaceGrVars::tags_list{}));

    CHECK_VARIABLES_APPROX(gsl::at(db::get<subcell_faces_gr_tag>(box), i),
                           expected_face_vars);
  }
}
