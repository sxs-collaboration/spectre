// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/BoundaryCorrectionTags.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/NeighborData.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/Reconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/NeighborData.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/NormalCovectorAndMagnitude.hpp"
#include "Evolution/DiscontinuousGalerkin/NormalVectorTags.hpp"
#include "Evolution/Systems/Burgers/BoundaryCorrections/BoundaryCorrection.hpp"
#include "Evolution/Systems/Burgers/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/Burgers/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/Burgers/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/Burgers/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Burgers/Fluxes.hpp"
#include "Evolution/Systems/Burgers/Subcell/NeighborPackagedData.hpp"
#include "Evolution/Systems/Burgers/System.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/Systems/Burgers/FiniteDifference/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Burgers {

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Burgers.Subcell.NeighborPackagedData",
                  "[Unit][Evolution]") {
  MAKE_GENERATOR(gen);

  // 1. create random U on an element and its neighbor elements
  // 2. send through reconstruction and compute FD fluxes on mortars
  // 3. feed argument variables of dg_package_data() function to
  //    the NeighborPackagedData struct and retrieve the packaged data
  // 4. check if it agrees with the expected value

  using evolved_vars_tags = typename System::variables_tag::tags_list;
  using fluxes_tags = typename Fluxes::return_tags;

  // Perform test with MC reconstruction & Rusanov riemann solver
  using reconstruction_used_for_test = fd::MonotisedCentral;
  using boundary_correction_used_for_test = BoundaryCorrections::Rusanov;

  // create an element and its neighbor elements
  DirectionMap<1, Neighbors<1>> element_neighbors{};
  for (size_t i = 0; i < 2; ++i) {
    element_neighbors[gsl::at(Direction<1>::all_directions(), i)] =
        Neighbors<1>{{ElementId<1>{i + 1, {}}}, {}};
  }
  const Element<1> element{ElementId<1>{0, {}}, element_neighbors};

  // generate random U on the dg mesh and project it to subcell mesh
  const size_t num_dg_pts = 5;
  const Mesh<1> dg_mesh{num_dg_pts, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const Mesh<1> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);

  Variables<evolved_vars_tags> volume_vars_dg{dg_mesh.number_of_grid_points()};
  std::uniform_real_distribution<> dist(-1.0, 1.0);
  fill_with_random_values(make_not_null(&volume_vars_dg), make_not_null(&gen),
                          make_not_null(&dist));
  Variables<evolved_vars_tags> volume_vars_subcell =
      evolution::dg::subcell::fd::project(volume_vars_dg, dg_mesh,
                                          subcell_mesh.extents());

  // generate (random) ghost data from neighbor
  auto logical_coords_subcell = logical_coordinates(subcell_mesh);
  const reconstruction_used_for_test reconstructor{};
  const auto compute_random_variable = [&gen, &dist](const auto& coords) {
    Variables<evolved_vars_tags> vars{get<0>(coords).size(), 0.0};
    fill_with_random_values(make_not_null(&vars), make_not_null(&gen),
                            make_not_null(&dist));
    return vars;
  };
  typename evolution::dg::subcell::Tags::
      NeighborDataForReconstructionAndRdmpTci<1>::type neighbor_data =
          TestHelpers::Burgers::fd::compute_neighbor_data(
              subcell_mesh, logical_coords_subcell, element.neighbors(),
              reconstructor.ghost_zone_size(), compute_random_variable);

  DirectionMap<1, std::optional<Variables<
                      tmpl::list<evolution::dg::Tags::MagnitudeOfNormal,
                                 evolution::dg::Tags::NormalCovector<1>>>>>
      normal_vectors{};
  for (const auto& direction : Direction<1>::all_directions()) {
    const auto coordinate_map =
        domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
            domain::CoordinateMaps::Identity<1>{});
    const auto moving_mesh_map =
        domain::make_coordinate_map<Frame::Grid, Frame::Inertial>(
            domain::CoordinateMaps::Identity<1>{});

    const Mesh<0> face_mesh = dg_mesh.slice_away(direction.dimension());
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
        unit_normal_vector_and_covector_and_magnitude<System>(
            make_not_null(&normal_vectors), make_not_null(&fields_on_face),
            direction, unnormalized_normal_covectors, moving_mesh_map);
  }

  auto box = db::create<db::AddSimpleTags<
      domain::Tags::Element<1>, domain::Tags::Mesh<1>,
      evolution::dg::subcell::Tags::Mesh<1>, typename System::variables_tag,
      evolution::dg::subcell::Tags::NeighborDataForReconstructionAndRdmpTci<1>,
      fd::Tags::Reconstructor, evolution::Tags::BoundaryCorrection<System>,
      domain::Tags::ElementMap<1, Frame::Grid>,
      domain::CoordinateMaps::Tags::CoordinateMap<1, Frame::Grid,
                                                  Frame::Inertial>,
      evolution::dg::subcell::Tags::Coordinates<1, Frame::ElementLogical>,
      domain::Tags::MeshVelocity<1>,
      evolution::dg::Tags::NormalCovectorAndMagnitude<1>>>(
      element, dg_mesh, subcell_mesh, volume_vars_dg, neighbor_data,
      std::unique_ptr<fd::Reconstructor>{
          std::make_unique<reconstruction_used_for_test>()},
      std::unique_ptr<BoundaryCorrections::BoundaryCorrection>{
          std::make_unique<boundary_correction_used_for_test>()},
      ElementMap<1, Frame::Grid>{
          ElementId<1>{0},
          domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(
              domain::CoordinateMaps::Identity<1>{})},
      domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
          domain::CoordinateMaps::Identity<1>{}),
      logical_coords_subcell,
      std::optional<tnsr::I<DataVector, 1, Frame::Inertial>>{}, normal_vectors);

  // Compute the packaged data
  std::vector<std::pair<Direction<1>, ElementId<1>>>
      mortars_to_reconstruct_to{};
  for (const auto& [direction, neighbors] : element.neighbors()) {
    mortars_to_reconstruct_to.emplace_back(direction, *neighbors.begin());
  }
  const auto packaged_data =
      subcell::NeighborPackagedData::apply(box, mortars_to_reconstruct_to);

  // Now for each directions, check that the packaged_data agrees with expected
  // values
  using dg_package_field_tags =
      typename boundary_correction_used_for_test::dg_package_field_tags;
  using dg_package_data_argument_tags =
      tmpl::append<evolved_vars_tags, fluxes_tags>;

  boundary_correction_used_for_test boundary_corr_for_test{};

  Variables<dg_package_data_argument_tags> vars_on_mortar_face{0};
  Variables<dg_package_field_tags> expected_fd_packaged_data_on_mortar{0};

  for (const auto& mortar_id : mortars_to_reconstruct_to) {
    const Direction<1>& direction = mortar_id.first;

    // Note : 1D mortar has only one face point
    vars_on_mortar_face.initialize(1);
    expected_fd_packaged_data_on_mortar.initialize(1);

    // reconstruct U on the mortar
    dynamic_cast<const reconstruction_used_for_test&>(reconstructor)
        .reconstruct_fd_neighbor(make_not_null(&vars_on_mortar_face),
                                 volume_vars_subcell, element, neighbor_data,
                                 subcell_mesh, direction);

    // compute fluxes
    Burgers::subcell::compute_fluxes(make_not_null(&vars_on_mortar_face));

    // revert normal vector
    auto normal_covector = get<evolution::dg::Tags::NormalCovector<1>>(
        *db::get<evolution::dg::Tags::NormalCovectorAndMagnitude<1>>(box).at(
            mortar_id.first));
    for (auto& t : normal_covector) {
      t *= -1.0;
    }

    // Compute the expected packaged data
    evolution::dg::Actions::detail::dg_package_data<System>(
        make_not_null(&expected_fd_packaged_data_on_mortar),
        boundary_corr_for_test, vars_on_mortar_face, normal_covector,
        {std::nullopt}, box,
        typename boundary_correction_used_for_test::
            dg_package_data_volume_tags{},
        dg_package_data_argument_tags{});

    std::vector<double> vector_to_check{
        expected_fd_packaged_data_on_mortar.data(),
        expected_fd_packaged_data_on_mortar.data() +
            expected_fd_packaged_data_on_mortar.size()};

    CHECK_ITERABLE_APPROX(vector_to_check, packaged_data.at(mortar_id));
  }
}
}  // namespace Burgers
