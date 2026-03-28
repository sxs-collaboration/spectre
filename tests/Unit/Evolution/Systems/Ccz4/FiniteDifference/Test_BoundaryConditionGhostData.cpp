// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataBox/MetavariablesTag.hpp"
#include "DataStructures/TaggedContainers.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Creators/Rectilinear.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Creators/Tags/ExternalBoundaryConditions.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/InterfaceLogicalCoordinates.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/DgSubcell/GhostZoneLogicalCoordinates.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/DirichletAnalytic.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/Sommerfeld.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/BoundaryConditionGhostData.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/DummyReconstructor.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Framework/Pypp.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Evolution/Systems/Ccz4/Solutions/Factory.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/CloneUniquePtrs.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4::fd {
namespace {

// Metavariables to parse the list of derived classes of boundary conditions
struct EvolutionMetaVars {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<BoundaryConditions::BoundaryCondition,
                             BoundaryConditions::standard_boundary_conditions>,
                  tmpl::pair<evolution::initial_data::InitialData,
                             Ccz4::Solutions::all_solutions>>;
  };
};

using SolutionForTest = Solutions::Ccz4WrappedGr<gr::Solutions::Minkowski<3>>;

using Vars = Variables<Ccz4::fd::Tags::spacetime_reconstruction_tags>;
Vars set_polynomial(const tnsr::I<DataVector, 3, Frame::Inertial>& coords,
                    const size_t max_degree) {
  DataVector result_vector{get<0>(coords).size(), 0.0};
  for (size_t degree_x = 0; degree_x <= max_degree; ++degree_x) {
    for (size_t degree_y = 0; degree_y <= max_degree - degree_x; ++degree_y) {
      for (size_t degree_z = 0; degree_z <= max_degree - degree_x - degree_y;
           ++degree_z) {
        result_vector += pow(get<0>(coords), degree_x) *
                         pow(get<1>(coords), degree_y) *
                         pow(get<2>(coords), degree_z);
      }
    }
  }
  Vars vars{get<0>(coords).size()};

  tmpl::for_each<Ccz4::fd::System::variables_tag_list>(
      [&]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
        for (auto& component : get<Tag>(vars)) {
          component = result_vector;
        }
      });

  return vars;
}

template <typename BoundaryConditionType>
void test(const BoundaryConditionType& boundary_condition,
          const SolutionForTest& solution) {
  const size_t num_dg_pts = 3;

  // Create a 3D element [-1, 1]^3 and use it for test
  const std::array<double, 3> lower_bounds{-1.0, -1.0, -1.0};
  const std::array<double, 3> upper_bounds{1.0, 1.0, 1.0};
  const std::array<size_t, 3> refinement_levels{0, 0, 0};
  const std::array<size_t, 3> number_of_grid_points{num_dg_pts, num_dg_pts,
                                                    num_dg_pts};
  const domain::creators::Brick brick(
      lower_bounds, upper_bounds, refinement_levels, number_of_grid_points,
      {{{{boundary_condition.get_clone(), boundary_condition.get_clone()}},
        {{boundary_condition.get_clone(), boundary_condition.get_clone()}},
        {{boundary_condition.get_clone(), boundary_condition.get_clone()}}}});
  auto domain = brick.domain();
  auto boundary_conditions = brick.external_boundary_conditions();
  const auto element = domain::create_initial_element(
      ElementId<3>{0, {SegmentId{0, 0}, SegmentId{0, 0}, SegmentId{0, 0}}},
      domain.blocks(), std::vector<std::array<size_t, 3>>{{refinement_levels}});

  // Mesh and coordinates
  const Mesh<3> dg_mesh{num_dg_pts, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const Mesh<3> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);

  using ReconstructorForTest = DummyReconstructor;
  const size_t ghost_zone_size{ReconstructorForTest{}.ghost_zone_size()};

  // Below are tags required by DirichletAnalytic boundary condition to compute
  // inertial coords of ghost FD cells:
  //  - time
  //  - functions of time
  //  - element map
  //  - coordinate map
  //  - subcell logical coordinates
  // -----------------------------------------

  const double time{0.5};

  const std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};

  const ElementMap<3, Frame::Grid> logical_to_grid_map(
      ElementId<3>{0},
      domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(
          domain::CoordinateMaps::Identity<3>{}));

  const auto grid_to_inertial_map =
      domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
          domain::CoordinateMaps::Identity<3>{});

  const auto subcell_logical_coords = logical_coordinates(subcell_mesh);

  // dummy neighbor data to put into DataBox
  typename evolution::dg::subcell::Tags::GhostDataForReconstruction<3>::type
      ghost_data{};

  const auto direction = Direction<3>::upper_xi();

  const auto ghost_logical_coords =
      evolution::dg::subcell::fd::ghost_zone_logical_coordinates(
          subcell_mesh, ghost_zone_size, direction);

  const auto ghost_inertial_coords = (*grid_to_inertial_map)(
      logical_to_grid_map(ghost_logical_coords), time, functions_of_time);

  const auto interior_inertial_coords = (*grid_to_inertial_map)(
      logical_to_grid_map(subcell_logical_coords), time, functions_of_time);

  // interior evolved variables for extrapolating into the ghost zone of
  // external boundary to compute derivatives for the Sommerfeld BC
  const Vars interior_evolved_vars =
      set_polynomial(interior_inertial_coords, 1);

  auto& [conformal_metric, conformal_factor, a_tilde, trace_extrinsic_curvature,
         theta, gamma_hat, lapse, shift, auxiliary_shift_b] =
      interior_evolved_vars;

  auto box = db::create<db::AddSimpleTags<
      Ccz4::Tags::ConformalMetric<DataVector, 3>, gr::Tags::Lapse<DataVector>,
      gr::Tags::Shift<DataVector, 3>, Ccz4::Tags::ConformalFactor<DataVector>,
      Ccz4::Tags::ATilde<DataVector, 3>,
      gr::Tags::TraceExtrinsicCurvature<DataVector>,
      Ccz4::Tags::Theta<DataVector>, Ccz4::Tags::GammaHat<DataVector, 3>,
      Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>,
      Parallel::Tags::MetavariablesImpl<EvolutionMetaVars>,
      domain::Tags::Domain<3>, domain::Tags::ExternalBoundaryConditions<3>,
      evolution::dg::subcell::Tags::Mesh<3>,
      evolution::dg::subcell::Tags::Coordinates<3, Frame::ElementLogical>,
      evolution::dg::subcell::Tags::GhostDataForReconstruction<3>,
      Ccz4::fd::Tags::Reconstructor, ::Tags::Time,
      domain::Tags::FunctionsOfTimeInitialize,
      domain::Tags::ElementMap<3, Frame::Grid>,
      domain::CoordinateMaps::Tags::CoordinateMap<3, Frame::Grid,
                                                  Frame::Inertial>>>(
      conformal_metric, lapse, shift, conformal_factor, a_tilde,
      trace_extrinsic_curvature, theta, gamma_hat, auxiliary_shift_b,
      EvolutionMetaVars{}, std::move(domain), std::move(boundary_conditions),
      subcell_mesh, subcell_logical_coords, ghost_data,
      std::unique_ptr<Ccz4::fd::Reconstructor>{
          std::make_unique<ReconstructorForTest>()},
      time, clone_unique_ptrs(functions_of_time),
      ElementMap<3, Frame::Grid>{
          ElementId<3>{0},
          domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(
              domain::CoordinateMaps::Identity<3>{})},
      domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
          domain::CoordinateMaps::Identity<3>{}));

  // compute FD ghost data and retrieve the result
  fd::BoundaryConditionGhostData::apply(make_not_null(&box), element,
                                        ReconstructorForTest{});
  const DirectionalId<3> mortar_id = {direction,
                                      ElementId<3>::external_boundary_id()};
  const DataVector& fd_ghost_data =
      get<evolution::dg::subcell::Tags::GhostDataForReconstruction<3>>(box)
          .at(mortar_id)
          .neighbor_ghost_data_for_reconstruction();

  const size_t num_face_pts{
      subcell_mesh.extents().slice_away(direction.dimension()).product()};
  Variables<System::variables_tag_list> ghost_zone_vars{num_face_pts *
                                                        ghost_zone_size};

  std::copy(fd_ghost_data.begin(),
            std::next(fd_ghost_data.begin(),
                      static_cast<std::ptrdiff_t>(ghost_zone_vars.size())),
            ghost_zone_vars.data());

  // now check values for each types of boundary conditions
  if (typeid(BoundaryConditionType) ==
      typeid(Ccz4::BoundaryConditions::DirichletAnalytic)) {
    const auto& expected_ghost_vars = solution.variables(
        ghost_inertial_coords, time, typename System::variables_tag_list{});
    tmpl::for_each<System::variables_tag_list>([&]<typename Tag>(
                                                   tmpl::type_<Tag> /*meta*/) {
      const std::string tag_name = db::tag_name<Tag>();
      CAPTURE(tag_name);
      CAPTURE(ghost_inertial_coords);
      CHECK(tuples::get<Tag>(expected_ghost_vars) == get<Tag>(ghost_zone_vars));
    });

  } else if (typeid(BoundaryConditionType) ==
             typeid(Ccz4::BoundaryConditions::Sommerfeld)) {
    const auto expected_ghost_vars = set_polynomial(ghost_inertial_coords, 1);
    tmpl::for_each<Ccz4::fd::System::variables_tag_list>(
        [&]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
          const std::string tag_name = db::tag_name<Tag>();
          CAPTURE(tag_name);
          for (auto expected_component = get<Tag>(expected_ghost_vars).cbegin(),
                    component = get<Tag>(ghost_zone_vars).cbegin();
               expected_component != get<Tag>(expected_ghost_vars).cend() and
               component != get<Tag>(ghost_zone_vars).cend();
               ++expected_component, ++component) {
            CHECK_ITERABLE_APPROX(*component, *expected_component);
          }
        });

  } else {
    FAIL("Boundary condition type not handled in test");
  }
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Ccz4.Fd.BCondGhostData",
                  "[Unit][Evolution]") {
  const SolutionForTest solution{};

  // test DirichletAnalytic BC
  test(TestHelpers::test_creation<Ccz4::BoundaryConditions::DirichletAnalytic,
                                  EvolutionMetaVars>("AnalyticPrescription:\n"
                                                     "  Ccz4(Minkowski)\n"),
       solution);
  test(Ccz4::BoundaryConditions::DirichletAnalytic{std::make_unique<
           Ccz4::Solutions::Ccz4WrappedGr<gr::Solutions::Minkowski<3>>>()},
       solution);

  // test Sommerfeld BC; solution is unused
  // test(Ccz4::BoundaryConditions::Sommerfeld{}, solution);

  // check that the periodic BC fails
#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      ([&solution]() {
        test(domain::BoundaryConditions::Periodic<
                 Ccz4::BoundaryConditions::BoundaryCondition>{},
             solution);
      })(),
      Catch::Matchers::ContainsSubstring("not on external boundaries"));
#endif
}
}  // namespace
}  // namespace Ccz4::fd
