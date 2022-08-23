// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Block.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Creators/Interval.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/InterfaceLogicalCoordinates.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/DgSubcell/GhostZoneLogicalCoordinates.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/SliceData.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/NeighborData.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/NormalCovectorAndMagnitude.hpp"
#include "Evolution/DiscontinuousGalerkin/NormalVectorTags.hpp"
#include "Evolution/Systems/Burgers/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/Burgers/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/Burgers/FiniteDifference/BoundaryConditionGhostData.hpp"
#include "Evolution/Systems/Burgers/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/Burgers/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/Burgers/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Burgers/System.hpp"
#include "Evolution/Systems/Burgers/Tags.hpp"
#include "Framework/Pypp.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Burgers/Linear.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Utilities/CloneUniquePtrs.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {

enum class TestCases { BcPointerIsNull, AllGoodWithDomain };

// Metavariables to parse the list of derived classes of boundary conditions
struct EvolutionMetaVars {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<Burgers::BoundaryConditions::BoundaryCondition,
                   Burgers::BoundaryConditions::standard_boundary_conditions>>;
  };
};

template <typename BoundaryConditionType>
void test(const BoundaryConditionType& boundary_condition,
          const TestCases test_this) {
  const size_t num_dg_pts = 3;

  // Create an 1D interval [-1, 1] and use it for test
  const std::array<double, 1> lower_x{-1.0};
  const std::array<double, 1> upper_x{1.0};
  const std::array<size_t, 1> refinement_level_x{0};
  const std::array<size_t, 1> number_of_grid_points_in_x{num_dg_pts};
  const auto interval = domain::creators::Interval(
      lower_x, upper_x, refinement_level_x, number_of_grid_points_in_x,
      std::make_unique<BoundaryConditionType>(boundary_condition),
      std::make_unique<BoundaryConditionType>(boundary_condition), nullptr);
  auto domain = interval.create_domain();
  const auto element = domain::Initialization::create_initial_element(
      ElementId<1>{0, {SegmentId{0, 0}}}, domain.blocks().at(0),
      std::vector<std::array<size_t, 1>>{{refinement_level_x}});

  // Mesh and coordinates
  const Mesh<1> dg_mesh{num_dg_pts, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const Mesh<1> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);

  // use MC reconstruction for test
  using ReconstructorForTest = Burgers::fd::MonotonisedCentral;

  // dummy neighbor data to put into DataBox
  typename evolution::dg::subcell::Tags::NeighborDataForReconstruction<1>::type
      neighbor_data{};

  // Below are tags required by DirichletAnalytic boundary condition to compute
  // inertial coords of ghost FD cells:
  //  - time
  //  - functions of time
  //  - element map
  //  - coordinate map
  //  - subcell logical coordinates
  //  - analytic solution

  const double time{0.0};

  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};

  const ElementMap<1, Frame::Grid> logical_to_grid_map(
      ElementId<1>{0},
      domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(
          domain::CoordinateMaps::Identity<1>{}));

  const auto grid_to_inertial_map =
      domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
          domain::CoordinateMaps::Identity<1>{});

  const auto subcell_logical_coords = logical_coordinates(subcell_mesh);

  using SolutionForTest = Burgers::Solutions::Linear;
  const SolutionForTest solution{-0.5};

  // Below are tags required by Outflow boundary condition
  //  - scalar field U on subcell mesh
  //  - mesh velocity
  //  - normal vectors

  const double volume_u_val = 1.0;
  Scalar<DataVector> u_subcell{subcell_mesh.number_of_grid_points(),
                               volume_u_val};

  std::optional<tnsr::I<DataVector, 1>> volume_mesh_velocity{};

  typename evolution::dg::Tags::NormalCovectorAndMagnitude<1>::type
      normal_vectors{};
  for (const auto& direction : Direction<1>::all_directions()) {
    const auto coordinate_map =
        domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
            domain::CoordinateMaps::Identity<1>{});
    const auto moving_mesh_map =
        domain::make_coordinate_map<Frame::Grid, Frame::Inertial>(
            domain::CoordinateMaps::Identity<1>{});

    const Mesh<0> face_mesh = subcell_mesh.slice_away(direction.dimension());
    const auto face_logical_coords =
        interface_logical_coordinates(face_mesh, direction);
    std::unordered_map<Direction<1>, tnsr::i<DataVector, 1, Frame::Inertial>>
        unnormalized_normal_covectors{};
    tnsr::i<DataVector, 1, Frame::Inertial> unnormalized_covector{};
    unnormalized_covector.get(0) =
        coordinate_map.inv_jacobian(face_logical_coords)
            .get(direction.dimension(), 0);
    unnormalized_normal_covectors[direction] = unnormalized_covector;
    Variables<tmpl::list<
        evolution::dg::Actions::detail::NormalVector<1>,
        evolution::dg::Actions::detail::OneOverNormalVectorMagnitude>>
        fields_on_face{face_mesh.number_of_grid_points()};

    normal_vectors[direction] = std::nullopt;
    evolution::dg::Actions::detail::
        unit_normal_vector_and_covector_and_magnitude<Burgers::System>(
            make_not_null(&normal_vectors), make_not_null(&fields_on_face),
            direction, unnormalized_normal_covectors, moving_mesh_map);
  }

  // create a box
  auto box = db::create<db::AddSimpleTags<
      Parallel::Tags::MetavariablesImpl<EvolutionMetaVars>,
      domain::Tags::Domain<1>, evolution::dg::subcell::Tags::Mesh<1>,
      evolution::dg::subcell::Tags::Coordinates<1, Frame::ElementLogical>,
      evolution::dg::subcell::Tags::NeighborDataForReconstruction<1>,
      Burgers::fd::Tags::Reconstructor, domain::Tags::MeshVelocity<1>,
      evolution::dg::Tags::NormalCovectorAndMagnitude<1>, Tags::Time,
      domain::Tags::FunctionsOfTimeInitialize,
      domain::Tags::ElementMap<1, Frame::Grid>,
      domain::CoordinateMaps::Tags::CoordinateMap<1, Frame::Grid,
                                                  Frame::Inertial>,
      Tags::AnalyticSolution<SolutionForTest>, Burgers::Tags::U>>(
      EvolutionMetaVars{}, std::move(domain), subcell_mesh,
      subcell_logical_coords, neighbor_data,
      std::unique_ptr<Burgers::fd::Reconstructor>{
          std::make_unique<ReconstructorForTest>()},
      volume_mesh_velocity, normal_vectors, time,
      clone_unique_ptrs(functions_of_time),
      ElementMap<1, Frame::Grid>{
          ElementId<1>{0},
          domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(
              domain::CoordinateMaps::Identity<1>{})},
      domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
          domain::CoordinateMaps::Identity<1>{}),
      solution, u_subcell);

  if (test_this == TestCases::BcPointerIsNull) {
#ifdef SPECTRE_DEBUG
    // replace the domain inside the box with the one without boundary
    // conditions.
    db::mutate<domain::Tags::Domain<1>>(
        make_not_null(&box),
        [&lower_x, &upper_x, &refinement_level_x, &number_of_grid_points_in_x](
            const gsl::not_null<Domain<1>*> domain_v) {
          const auto interval_without_bc = domain::creators::Interval(
              lower_x, upper_x, refinement_level_x, number_of_grid_points_in_x,
              nullptr, nullptr, nullptr);
          *domain_v = interval_without_bc.create_domain();
        });
    // See if the ASSERT catches this error correctly.
    CHECK_THROWS_WITH(
        ([&box, &element]() {
          Burgers::fd::BoundaryConditionGhostData::apply(
              make_not_null(&box), element, ReconstructorForTest{});
        })(),
        Catch::Contains("Boundary condition is not set (the pointer is null)"));
#endif

  } else if (test_this == TestCases::AllGoodWithDomain) {
    // compute FD ghost data and retrieve the result
    Burgers::fd::BoundaryConditionGhostData::apply(make_not_null(&box), element,
                                                   ReconstructorForTest{});
    const auto direction = Direction<1>::upper_xi();
    const std::pair mortar_id = {direction,
                                 ElementId<1>::external_boundary_id()};
    const std::vector<double>& fd_ghost_data =
        get<evolution::dg::subcell::Tags::NeighborDataForReconstruction<1>>(box)
            .at(mortar_id);

    // now check values for each types of boundary conditions

    if (typeid(BoundaryConditionType) ==
        typeid(Burgers::BoundaryConditions::Dirichlet)) {
      const std::vector<double> expected_ghost_u(fd_ghost_data.size(), 0.3);
      CHECK_ITERABLE_APPROX(expected_ghost_u, fd_ghost_data);
    }

    if (typeid(BoundaryConditionType) ==
        typeid(Burgers::BoundaryConditions::DirichletAnalytic)) {
      const size_t ghost_zone_size{ReconstructorForTest{}.ghost_zone_size()};

      const auto ghost_logical_coords =
          evolution::dg::subcell::fd::ghost_zone_logical_coordinates(
              subcell_mesh, ghost_zone_size, direction);

      const auto ghost_inertial_coords = (*grid_to_inertial_map)(
          logical_to_grid_map(ghost_logical_coords), time, functions_of_time);

      const auto solution_py = pypp::call<Scalar<DataVector>>(
          "Linear", "u", ghost_inertial_coords, time);
      std::vector<double> expected_ghost_u{
          get(solution_py).data(),
          get(solution_py).data() + get(solution_py).size()};

      CHECK_ITERABLE_APPROX(expected_ghost_u, fd_ghost_data);
    }

    if (typeid(BoundaryConditionType) ==
        typeid(Burgers::BoundaryConditions::Outflow)) {
      // U = 1.0 on the subcell mesh, so the outflow condition will not throw
      // any error. Here we just check if
      // `Burgers::fd::BoundaryConditionGhostData::apply()` has correctly filled
      // out `fd_ghost_data` with the outermost value.
      const std::vector<double> expected_ghost_u(fd_ghost_data.size(),
                                                 volume_u_val);
      CHECK_ITERABLE_APPROX(expected_ghost_u, fd_ghost_data);

      // Test when U=-1.0, which will raise ERROR by violating the outflow
      // condition.
      db::mutate<Burgers::Tags::U>(
          make_not_null(&box),
          [](const gsl::not_null<Scalar<DataVector>*> volume_u) {
            get(*volume_u) = -1.0;
          });
      // See if the code fails correctly.
      CHECK_THROWS_WITH(
          ([&box, &element]() {
            Burgers::fd::BoundaryConditionGhostData::apply(
                make_not_null(&box), element, ReconstructorForTest{});
          })(),
          Catch::Contains("Outflow boundary condition (subcell) violated"));

      // Test when the volume mesh velocity has value, which will raise ERROR.
      db::mutate<domain::Tags::MeshVelocity<1>>(
          make_not_null(&box),
          [&subcell_mesh](
              const gsl::not_null<std::optional<tnsr::I<DataVector, 1>>*>
                  mesh_velocity) {
            tnsr::I<DataVector, 1> volume_mesh_velocity_tnsr(
                subcell_mesh.number_of_grid_points(), 0.0);
            *mesh_velocity = volume_mesh_velocity_tnsr;
          });
      // See if the code fails correctly.
      CHECK_THROWS_WITH(
          ([&box, &element]() {
            Burgers::fd::BoundaryConditionGhostData::apply(
                make_not_null(&box), element, ReconstructorForTest{});
          })(),
          Catch::Contains("Subcell currently does not support moving mesh"));
    }
  }
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Burgers.Fd.BCondGhostData",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/Burgers/FiniteDifference"};

  for (const auto test_this :
       {TestCases::BcPointerIsNull, TestCases::AllGoodWithDomain}) {
    test(TestHelpers::test_creation<Burgers::BoundaryConditions::Dirichlet>(
             "U: 0.3\n"),
         test_this);
    test(Burgers::BoundaryConditions::DirichletAnalytic{}, test_this);
    test(Burgers::BoundaryConditions::Outflow{}, test_this);
  }

  // check that the periodic BC fails
#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(([]() {
                      test(
                          domain::BoundaryConditions::Periodic<
                              Burgers::BoundaryConditions::BoundaryCondition>{},
                          TestCases::AllGoodWithDomain);
                    })(),
                    Catch::Contains("not on external boundaries"));
#endif
}
}  // namespace
