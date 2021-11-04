// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Evolution/BoundaryCorrectionTags.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/OnSubcellFaces.hpp"
#include "Evolution/Initialization/Tags.hpp"
#include "Evolution/Systems/ScalarAdvection/BoundaryCorrections/BoundaryCorrection.hpp"
#include "Evolution/Systems/ScalarAdvection/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/ScalarAdvection/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/ScalarAdvection/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/ScalarAdvection/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/ScalarAdvection/Subcell/TimeDerivative.hpp"
#include "Evolution/Systems/ScalarAdvection/Subcell/VelocityAtFace.hpp"
#include "Evolution/Systems/ScalarAdvection/System.hpp"
#include "Evolution/Systems/ScalarAdvection/Tags.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/Systems/ScalarAdvection/FiniteDifference/TestHelpers.hpp"
#include "Utilities/CloneUniquePtrs.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarAdvection {
namespace {
template <size_t Dim>
void test_subcell_timederivative() {
  using evolved_vars_tag = typename System<Dim>::variables_tag;
  using dt_variables_tag = db::add_tag_prefix<::Tags::dt, evolved_vars_tag>;

  using velocity_field = Tags::VelocityField<Dim>;
  using subcell_velocity_field =
      evolution::dg::subcell::Tags::Inactive<velocity_field>;
  using subcell_faces_velocity_field =
      evolution::dg::subcell::Tags::OnSubcellFaces<velocity_field, Dim>;

  DirectionMap<Dim, Neighbors<Dim>> neighbors{};
  for (size_t i = 0; i < 2 * Dim; ++i) {
    neighbors[gsl::at(Direction<Dim>::all_directions(), i)] =
        Neighbors<Dim>{{ElementId<Dim>{i + 1, {}}}, {}};
  }
  const Element<Dim> element{ElementId<Dim>{0, {}}, neighbors};

  const size_t num_dg_pts_per_dimension = 5;
  const Mesh<Dim> dg_mesh{num_dg_pts_per_dimension, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
  const Mesh<Dim> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);

  // Perform test with MC reconstruction & Rusanov riemann solver
  using ReconstructionForTest = typename fd::MonotisedCentral<Dim>;
  using BoundaryCorrectionForTest = typename BoundaryCorrections::Rusanov<Dim>;

  // required for calling ScalarAdvection::VelocityAtFace::apply()
  const double time{0.0};
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};

  // Set the testing profile for the scalar field U.
  // Here we use
  //   * U(x)   = 2x + 1        (for 1D)
  //   * U(x,y) = 2x + y^2 + 1  (for 2D)
  const auto compute_test_solution = [](const auto& coords) {
    using tag = Tags::U;
    Variables<tmpl::list<tag>> vars{get<0>(coords).size(), 0.0};
    get(get<tag>(vars)) += 2.0 * coords.get(0) + 1.0;
    if constexpr (Dim == 2) {
      get(get<tag>(vars)) += square(coords.get(1));
    }
    return vars;
  };
  auto logical_coords_subcell = logical_coordinates(subcell_mesh);
  const auto volume_vars_subcell =
      compute_test_solution(logical_coords_subcell);

  // set the ghost data from neighbor
  const ReconstructionForTest reconstructor{};
  typename evolution::dg::subcell::Tags::
      NeighborDataForReconstructionAndRdmpTci<Dim>::type neighbor_data =
          TestHelpers::ScalarAdvection::fd::compute_neighbor_data(
              subcell_mesh, logical_coords_subcell, element.neighbors(),
              reconstructor.ghost_zone_size(), compute_test_solution);

  auto box = db::create<db::AddSimpleTags<
      domain::Tags::Element<Dim>, evolution::dg::subcell::Tags::Mesh<Dim>,
      evolved_vars_tag, dt_variables_tag,
      evolution::dg::subcell::Tags::NeighborDataForReconstructionAndRdmpTci<
          Dim>,
      fd::Tags::Reconstructor<Dim>,
      evolution::Tags::BoundaryCorrection<System<Dim>>,
      Initialization::Tags::InitialTime, domain::Tags::FunctionsOfTime,
      domain::Tags::ElementMap<Dim, Frame::Grid>,
      domain::CoordinateMaps::Tags::CoordinateMap<Dim, Frame::Grid,
                                                  Frame::Inertial>,
      evolution::dg::subcell::Tags::Coordinates<Dim, Frame::ElementLogical>,
      subcell_velocity_field, subcell_faces_velocity_field,
      evolution::dg::Tags::MortarData<Dim>>>(
      element, subcell_mesh, volume_vars_subcell,
      Variables<typename dt_variables_tag::tags_list>{
          subcell_mesh.number_of_grid_points()},
      neighbor_data,
      std::unique_ptr<fd::Reconstructor<Dim>>{
          std::make_unique<ReconstructionForTest>()},
      std::unique_ptr<BoundaryCorrections::BoundaryCorrection<Dim>>{
          std::make_unique<BoundaryCorrectionForTest>()},
      time, clone_unique_ptrs(functions_of_time),
      ElementMap<Dim, Frame::Grid>{
          ElementId<Dim>{0},
          domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(
              domain::CoordinateMaps::Identity<Dim>{})},
      domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
          domain::CoordinateMaps::Identity<Dim>{}),
      logical_coords_subcell, typename subcell_velocity_field::type{},
      std::array<typename subcell_faces_velocity_field::type::value_type,
                 Dim>{},
      typename evolution::dg::Tags::MortarData<Dim>::type{});

  // Compute face-centered velocity field and add it to the box. This action
  // needs to be called in prior since TimeDerivative::apply() internally
  // retrieves face-centered values of velocity field when packaging data for
  // riemann solve.
  db::mutate_apply<ScalarAdvection::subcell::VelocityAtFace<Dim>>(
      make_not_null(&box));

  {
    const auto coordinate_map =
        domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
            domain::CoordinateMaps::Identity<Dim>{});
    InverseJacobian<DataVector, Dim, Frame::ElementLogical, Frame::Grid>
        cell_centered_logical_to_grid_inv_jacobian{};
    const auto cell_centered_logical_to_inertial_inv_jacobian =
        coordinate_map.inv_jacobian(logical_coords_subcell);
    for (size_t i = 0; i < cell_centered_logical_to_grid_inv_jacobian.size();
         ++i) {
      cell_centered_logical_to_grid_inv_jacobian[i] =
          cell_centered_logical_to_inertial_inv_jacobian[i];
    }
    subcell::TimeDerivative<Dim>::apply(
        make_not_null(&box), cell_centered_logical_to_grid_inv_jacobian,
        determinant(cell_centered_logical_to_grid_inv_jacobian));
  }

  const auto& dt_vars = db::get<dt_variables_tag>(box);

  // Analytic time derivative of U for the testing profile
  //   * dt(U) = -2          (for 1D)
  //   * dt(U) = -1 +3y -2xy (for 2D)
  const auto compute_test_derivative = [](const auto& coords) {
    using tag = ::Tags::dt<Tags::U>;
    Variables<tmpl::list<tag>> dt_expected{get<0>(coords).size(), 0.0};
    if constexpr (Dim == 1) {
      get(get<tag>(dt_expected)) += -2.0;
    } else if constexpr (Dim == 2) {
      get(get<tag>(dt_expected)) +=
          -1.0 + 3.0 * coords.get(1) - 2.0 * coords.get(0) * coords.get(1);
    }
    return dt_expected;
  };

  CHECK_ITERABLE_APPROX(get<::Tags::dt<Tags::U>>(dt_vars),
                        get<::Tags::dt<Tags::U>>(
                            compute_test_derivative(logical_coords_subcell)));
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.ScalarAdvection.Subcell.TimeDerivative",
    "[Unit][Evolution]") {
  test_subcell_timederivative<1>();
  test_subcell_timederivative<2>();
}
}  // namespace ScalarAdvection
