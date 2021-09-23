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
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
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
#include "Evolution/DgSubcell/Tags/OnSubcellFaces.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/NormalCovectorAndMagnitude.hpp"
#include "Evolution/DiscontinuousGalerkin/NormalVectorTags.hpp"
#include "Evolution/Initialization/Tags.hpp"
#include "Evolution/Systems/ScalarAdvection/BoundaryCorrections/BoundaryCorrection.hpp"
#include "Evolution/Systems/ScalarAdvection/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/ScalarAdvection/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/ScalarAdvection/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/ScalarAdvection/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/ScalarAdvection/Fluxes.hpp"
#include "Evolution/Systems/ScalarAdvection/Subcell/NeighborPackagedData.hpp"
#include "Evolution/Systems/ScalarAdvection/Subcell/VelocityAtFace.hpp"
#include "Evolution/Systems/ScalarAdvection/System.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/Systems/ScalarAdvection/FiniteDifference/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/CloneUniquePtrs.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarAdvection {
namespace {
template <size_t Dim>
auto compute_face_tensor() {}

template <size_t Dim>
void test_neighbor_packaged_data(const size_t num_dg_pts_per_dimension,
                                 const gsl::not_null<std::mt19937*> gen) {
  // 1. create random U on an element and its neighbor elements
  // 2. send through reconstruction and compute FD fluxes on mortars
  // 3. feed argument variables of dg_package_data() function to
  //    the NeighborPackagedData struct and retrieve the packaged data
  // 4. check if it agrees with the expected value

  using evolved_vars_tags = typename System<Dim>::variables_tag::tags_list;
  using fluxes_tags = typename Fluxes<Dim>::return_tags;

  using velocity_field = Tags::VelocityField<Dim>;
  using subcell_velocity_field =
      evolution::dg::subcell::Tags::Inactive<velocity_field>;
  using subcell_faces_velocity_field =
      evolution::dg::subcell::Tags::OnSubcellFaces<velocity_field, Dim>;

  // Perform test with MC reconstruction & Rusanov riemann solver
  using ReconstructionForTest = typename fd::MonotisedCentral<Dim>;
  using BoundaryCorrectionForTest = typename BoundaryCorrections::Rusanov<Dim>;

  // create an element and its neighbor elements
  DirectionMap<Dim, Neighbors<Dim>> element_neighbors{};
  for (size_t i = 0; i < 2 * Dim; ++i) {
    element_neighbors[gsl::at(Direction<Dim>::all_directions(), i)] =
        Neighbors<Dim>{{ElementId<Dim>{i + 1, {}}}, {}};
  }
  const Element<Dim> element{ElementId<Dim>{0, {}}, element_neighbors};

  // generate random U on the dg mesh and project it to subcell mesh
  const Mesh<Dim> dg_mesh{num_dg_pts_per_dimension, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
  const Mesh<Dim> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);

  Variables<evolved_vars_tags> volume_vars_dg{dg_mesh.number_of_grid_points()};
  std::uniform_real_distribution<> dist(-1.0, 1.0);
  fill_with_random_values(make_not_null(&volume_vars_dg), gen,
                          make_not_null(&dist));
  Variables<evolved_vars_tags> volume_vars_subcell =
      evolution::dg::subcell::fd::project(volume_vars_dg, dg_mesh,
                                          subcell_mesh.extents());

  // below are required for calling ScalarAdvection::VelocityAtFace::apply()
  // function to compute velocity field on interfaces
  const double time{0.0};
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};

  // generate (random) ghost data from neighbor
  auto logical_coords_subcell = logical_coordinates(subcell_mesh);
  const ReconstructionForTest reconstructor{};
  const auto compute_random_variable = [&gen, &dist](const auto& coords) {
    Variables<evolved_vars_tags> vars{get<0>(coords).size(), 0.0};
    fill_with_random_values(make_not_null(&vars), gen, make_not_null(&dist));
    return vars;
  };
  typename evolution::dg::subcell::Tags::
      NeighborDataForReconstructionAndRdmpTci<Dim>::type neighbor_data =
          TestHelpers::ScalarAdvection::fd::compute_neighbor_data(
              subcell_mesh, logical_coords_subcell, element.neighbors(),
              reconstructor.ghost_zone_size(), compute_random_variable);

  DirectionMap<Dim, std::optional<Variables<
                        tmpl::list<evolution::dg::Tags::MagnitudeOfNormal,
                                   evolution::dg::Tags::NormalCovector<Dim>>>>>
      normal_vectors{};
  for (const auto& direction : Direction<Dim>::all_directions()) {
    const auto coordinate_map =
        domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
            domain::CoordinateMaps::Identity<Dim>{});
    const auto moving_mesh_map =
        domain::make_coordinate_map<Frame::Grid, Frame::Inertial>(
            domain::CoordinateMaps::Identity<Dim>{});

    const Mesh<Dim - 1> face_mesh = dg_mesh.slice_away(direction.dimension());
    const auto face_logical_coords =
        interface_logical_coordinates(face_mesh, direction);
    std::unordered_map<Direction<Dim>,
                       tnsr::i<DataVector, Dim, Frame::Inertial>>
        unnormalized_normal_covectors{};
    tnsr::i<DataVector, Dim, Frame::Inertial> unnormalized_covector{};
    for (size_t i = 0; i < Dim; ++i) {
      unnormalized_covector.get(i) =
          coordinate_map.inv_jacobian(face_logical_coords)
              .get(direction.dimension(), i);
    }
    unnormalized_normal_covectors[direction] = unnormalized_covector;
    Variables<tmpl::list<
        evolution::dg::Actions::detail::NormalVector<Dim>,
        evolution::dg::Actions::detail::OneOverNormalVectorMagnitude>>
        fields_on_face{face_mesh.number_of_grid_points()};

    normal_vectors[direction] = std::nullopt;
    evolution::dg::Actions::detail::
        unit_normal_vector_and_covector_and_magnitude<System<Dim>>(
            make_not_null(&normal_vectors), make_not_null(&fields_on_face),
            direction, unnormalized_normal_covectors, moving_mesh_map);
  }

  auto box = db::create<db::AddSimpleTags<
      domain::Tags::Element<Dim>, domain::Tags::Mesh<Dim>,
      evolution::dg::subcell::Tags::Mesh<Dim>,
      typename System<Dim>::variables_tag,
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
      domain::Tags::MeshVelocity<Dim>,
      evolution::dg::Tags::NormalCovectorAndMagnitude<Dim>>>(
      element, dg_mesh, subcell_mesh, volume_vars_dg, neighbor_data,
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
      std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>{},
      normal_vectors);

  // Compute face-centered velocity field and add it to the box. This action
  // needs to be called in prior since NeighborPackagedData::apply() internally
  // retrieves face-centered values of velocity field when computing fluxes.
  db::mutate_apply<ScalarAdvection::subcell::VelocityAtFace<Dim>>(
      make_not_null(&box));

  // Compute the packaged data
  std::vector<std::pair<Direction<Dim>, ElementId<Dim>>>
      mortars_to_reconstruct_to{};
  for (const auto& [direction, neighbors] : element.neighbors()) {
    mortars_to_reconstruct_to.emplace_back(direction, *neighbors.begin());
  }
  const auto packaged_data =
      subcell::NeighborPackagedData<Dim>::apply(box, mortars_to_reconstruct_to);

  // Now for each directions, check that the packaged_data agrees with expected
  // values
  using dg_package_field_tags =
      typename BoundaryCorrectionForTest ::dg_package_field_tags;
  using dg_package_data_argument_tags = tmpl::append<
      evolved_vars_tags, fluxes_tags,
      typename BoundaryCorrectionForTest ::dg_package_data_temporary_tags>;

  BoundaryCorrectionForTest boundary_corr_for_test{};

  Variables<dg_package_data_argument_tags> vars_on_mortar_face{0};
  Variables<dg_package_field_tags> expected_fd_packaged_data_on_mortar{0};

  for (const auto& mortar_id : mortars_to_reconstruct_to) {
    const Direction<Dim>& direction = mortar_id.first;
    Index<Dim> extents = subcell_mesh.extents();

    if constexpr (Dim == 1) {
      // Note : 1D mortar has only one face point
      vars_on_mortar_face.initialize(1);
      expected_fd_packaged_data_on_mortar.initialize(1);
    } else {
      ++extents[direction.dimension()];
      const size_t num_mortar_face_pts =
          subcell_mesh.extents().slice_away(direction.dimension()).product();
      vars_on_mortar_face.initialize(num_mortar_face_pts);
      expected_fd_packaged_data_on_mortar.initialize(num_mortar_face_pts);
    }

    // reconstruct U on the mortar
    dynamic_cast<const ReconstructionForTest&>(reconstructor)
        .reconstruct_fd_neighbor(make_not_null(&vars_on_mortar_face),
                                 volume_vars_subcell, element, neighbor_data,
                                 subcell_mesh, direction);

    // retrieve face-centered velocity field and slice it on the mortar, then
    // compute fluxes with it
    const auto& velocity_field_on_mortar = db::get<
        evolution::dg::subcell::Tags::OnSubcellFaces<velocity_field, Dim>>(box);
    data_on_slice(make_not_null(&get<velocity_field>(vars_on_mortar_face)),
                  gsl::at(velocity_field_on_mortar, direction.dimension()),
                  extents, direction.dimension(),
                  direction.side() == Side::Lower
                      ? 0
                      : extents[direction.dimension()] - 1);
    ScalarAdvection::subcell::compute_fluxes<Dim>(
        make_not_null(&vars_on_mortar_face));

    // reverse normal vector
    auto normal_covector = get<evolution::dg::Tags::NormalCovector<Dim>>(
        *db::get<evolution::dg::Tags::NormalCovectorAndMagnitude<Dim>>(box).at(
            mortar_id.first));
    for (auto& t : normal_covector) {
      t *= -1.0;
    }

    // Note : need to do the projection in 2d
    if constexpr (Dim > 1) {
      const auto dg_normal_covector = normal_covector;
      for (size_t i = 0; i < Dim; ++i) {
        normal_covector.get(i) = evolution::dg::subcell::fd::project(
            dg_normal_covector.get(i),
            dg_mesh.slice_away(mortar_id.first.dimension()),
            subcell_mesh.extents().slice_away(mortar_id.first.dimension()));
      }
    }

    // Compute the expected packaged data
    evolution::dg::Actions::detail::dg_package_data<System<Dim>>(
        make_not_null(&expected_fd_packaged_data_on_mortar),
        boundary_corr_for_test, vars_on_mortar_face, normal_covector,
        {std::nullopt}, box,
        typename BoundaryCorrectionForTest ::dg_package_data_volume_tags{},
        dg_package_data_argument_tags{});

    if constexpr (Dim == 1) {
      // no need to reconstruct back to DG grid for 1D
      std::vector<double> vector_to_check{
          expected_fd_packaged_data_on_mortar.data(),
          expected_fd_packaged_data_on_mortar.data() +
              expected_fd_packaged_data_on_mortar.size()};

      CHECK_ITERABLE_APPROX(vector_to_check, packaged_data.at(mortar_id));
    } else {
      // reconstruct from FD to DG grid for 2d
      const auto expected_dg_packaged_data =
          evolution::dg::subcell::fd::reconstruct(
              expected_fd_packaged_data_on_mortar,
              dg_mesh.slice_away(mortar_id.first.dimension()),
              subcell_mesh.extents().slice_away(mortar_id.first.dimension()));

      std::vector<double> vector_to_check{
          expected_dg_packaged_data.data(),
          expected_dg_packaged_data.data() + expected_dg_packaged_data.size()};

      CHECK_ITERABLE_APPROX(vector_to_check, packaged_data.at(mortar_id));
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.ScalarAdvection.Subcell.NeighborPackagedData",
    "[Unit][Evolution]") {
  const size_t num_dg_pts_per_dimension = 5;
  MAKE_GENERATOR(gen);

  test_neighbor_packaged_data<1>(num_dg_pts_per_dimension, make_not_null(&gen));
  test_neighbor_packaged_data<2>(num_dg_pts_per_dimension, make_not_null(&gen));
}
}  // namespace ScalarAdvection
