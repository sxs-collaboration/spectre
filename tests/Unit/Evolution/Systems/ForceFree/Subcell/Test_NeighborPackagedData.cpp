// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Slice.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/InterfaceLogicalCoordinates.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/Side.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/BoundaryCorrectionTags.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/Reconstruction.hpp"
#include "Evolution/DgSubcell/ReconstructionMethod.hpp"
#include "Evolution/DgSubcell/ReconstructionOrder.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/OnSubcellFaces.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/NormalCovectorAndMagnitude.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/PackageDataImpl.hpp"
#include "Evolution/DiscontinuousGalerkin/NormalVectorTags.hpp"
#include "Evolution/Systems/ForceFree/BoundaryCorrections/Rusanov.hpp"
#include "Evolution/Systems/ForceFree/FiniteDifference/MonotonisedCentral.hpp"
#include "Evolution/Systems/ForceFree/FiniteDifference/ReconstructWork.hpp"
#include "Evolution/Systems/ForceFree/FiniteDifference/ReconstructWork.tpp"
#include "Evolution/Systems/ForceFree/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/ForceFree/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/ForceFree/Fluxes.hpp"
#include "Evolution/Systems/ForceFree/Subcell/ComputeFluxes.hpp"
#include "Evolution/Systems/ForceFree/Subcell/NeighborPackagedData.hpp"
#include "Evolution/Systems/ForceFree/System.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/Systems/ForceFree/FiniteDifference/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/AnalyticSolutions/ForceFree/FastWave.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/CloneUniquePtrs.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"

namespace ForceFree {
namespace {

void test_neighbor_packaged_data(const gsl::not_null<std::mt19937*> gen) {
  //
  // Test is performed as follows
  //
  // 1. create random U on an element and its neighbor elements
  // 2. send through subcell reconstruction and compute FD fluxes on mortars
  // 3. feed argument variables of dg_package_data() function to
  //    the NeighborPackagedData struct and retrieve the packaged data
  // 4. check if it agrees with expected values
  //

  using evolved_vars_tags = typename System::variables_tag::tags_list;
  using fluxes_tags = typename Fluxes::return_tags;

  using ReconstructionForTest = fd::MonotonisedCentral;
  using BoundaryCorrectionForTest = BoundaryCorrections::Rusanov;

  using SolutionForTest = Solutions::FastWave;
  const SolutionForTest solution{};

  const double time{0.0};
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};

  // create an element and its neighbor elements
  DirectionMap<3, Neighbors<3>> element_neighbors{};
  for (size_t i = 0; i < 2 * 3_st; ++i) {
    element_neighbors[gsl::at(Direction<3>::all_directions(), i)] =
        Neighbors<3>{{ElementId<3>{i + 1, {}}}, {}};
  }
  const Element<3> element{ElementId<3>{0, {}}, element_neighbors};

  // Mesh (DG, Subcell), coordinate maps, logical and inertial coords.
  const size_t num_dg_pts_per_dimension = 5;
  const Mesh<3> dg_mesh{num_dg_pts_per_dimension, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const Mesh<3> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);

  const auto logical_to_grid_map = ElementMap<3, Frame::Grid>{
      ElementId<3>{0},
      domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(
          domain::CoordinateMaps::Identity<3>{})};
  const auto grid_to_inertial_map =
      domain::make_coordinate_map<Frame::Grid, Frame::Inertial>(
          domain::CoordinateMaps::Identity<3>{});

  const auto dg_logical_coords = logical_coordinates(dg_mesh);
  const auto dg_inertial_coords = grid_to_inertial_map(
      logical_to_grid_map(dg_logical_coords), time, functions_of_time);
  const auto subcell_logical_coords = logical_coordinates(subcell_mesh);
  const auto subcell_inertial_coords = grid_to_inertial_map(
      logical_to_grid_map(subcell_logical_coords), time, functions_of_time);

  // Create background metric variables on DG mesh and subcell mesh
  Variables<typename System::spacetime_variables_tag::tags_list> dg_gr_vars{
      dg_mesh.number_of_grid_points()};
  dg_gr_vars.assign_subset(solution.variables(
      dg_inertial_coords, time,
      typename System::spacetime_variables_tag::tags_list{}));
  using inactive_metric_vars_tags =
      evolution::dg::subcell::Tags::Inactive<System::spacetime_variables_tag>;
  inactive_metric_vars_tags::type subcell_gr_vars{
      subcell_mesh.number_of_grid_points()};
  subcell_gr_vars.assign_subset(solution.variables(
      subcell_inertial_coords, time,
      typename System::spacetime_variables_tag::tags_list{}));

  // Compute face-centered subcell GR vars
  std::array<typename System::flux_spacetime_variables_tag::type, 3>
      face_centered_gr_vars{};
  {
    const size_t num_face_centered_mesh_grid_pts =
        (subcell_mesh.extents(0) + 1) * subcell_mesh.extents(1) *
        subcell_mesh.extents(2);
    for (size_t d = 0; d < 3; ++d) {
      gsl::at(face_centered_gr_vars, d)
          .initialize(num_face_centered_mesh_grid_pts);
      const auto basis = make_array<3>(subcell_mesh.basis(0));
      auto quadrature = make_array<3>(subcell_mesh.quadrature(0));
      auto extents = make_array<3>(subcell_mesh.extents(0));
      gsl::at(extents, d) = subcell_mesh.extents(0) + 1;
      gsl::at(quadrature, d) = Spectral::Quadrature::FaceCentered;
      const Mesh<3> face_centered_mesh{extents, basis, quadrature};
      const auto face_centered_logical_coords =
          logical_coordinates(face_centered_mesh);
      const auto face_centered_inertial_coords = grid_to_inertial_map(
          logical_to_grid_map(face_centered_logical_coords), time,
          functions_of_time);
      face_centered_gr_vars.at(d).assign_subset(solution.variables(
          face_centered_inertial_coords, time,
          typename System::flux_spacetime_variables_tag::tags_list{}));
    }
  }

  // generate random volume vars on the dg mesh and project it to subcell mesh
  Variables<evolved_vars_tags> dg_evolved_vars{dg_mesh.number_of_grid_points()};
  std::uniform_real_distribution<> dist(-1.0, 1.0);
  fill_with_random_values(make_not_null(&dg_evolved_vars), gen,
                          make_not_null(&dist));
  Variables<evolved_vars_tags> volume_vars_subcell =
      evolution::dg::subcell::fd::project(dg_evolved_vars, dg_mesh,
                                          subcell_mesh.extents());

  // We also need TildeJ in the DataBox when performing FD reconstruction.
  // For testing purpose it is okay to have TildeJ on DG grid not necessarily
  // consistent with DG volume variables, but we need correctly projected values
  // on FD grid.
  const auto dg_tilde_j = make_with_random_values<Tags::TildeJ::type>(
      gen, make_not_null(&dist), dg_mesh.number_of_grid_points());
  const auto subcell_tilde_j = [&dg_tilde_j, &dg_mesh, &subcell_mesh]() {
    Variables<tmpl::list<Tags::TildeJ>> var{dg_mesh.number_of_grid_points()};
    for (size_t d = 0; d < 3; ++d) {
      get<Tags::TildeJ>(var).get(d) = dg_tilde_j.get(d);
    }
    return get<Tags::TildeJ>(evolution::dg::subcell::fd::project(
        var, dg_mesh, subcell_mesh.extents()));
  }();

  // generate random ghost data in neighboring elements
  const ReconstructionForTest reconstructor{};
  const auto compute_random_variable = [&gen, &dist](const auto& coords) {
    Variables<ForceFree::fd::tags_list_for_reconstruction> vars{
        get<0>(coords).size(), 0.0};
    fill_with_random_values(make_not_null(&vars), gen, make_not_null(&dist));
    return vars;
  };
  typename evolution::dg::subcell::Tags::GhostDataForReconstruction<3>::type
      ghost_data = TestHelpers::ForceFree::fd::compute_ghost_data(
          subcell_mesh, subcell_logical_coords, element.neighbors(),
          reconstructor.ghost_zone_size(), compute_random_variable);

  // Assign normal vector
  DirectionMap<3, std::optional<Variables<
                      tmpl::list<evolution::dg::Tags::MagnitudeOfNormal,
                                 evolution::dg::Tags::NormalCovector<3>>>>>
      normal_vectors{};
  for (const auto& direction : Direction<3>::all_directions()) {
    using inverse_spatial_metric_tag =
        typename System::inverse_spatial_metric_tag;
    const Mesh<2> face_mesh = dg_mesh.slice_away(direction.dimension());
    const auto face_logical_coords =
        interface_logical_coordinates(face_mesh, direction);
    std::unordered_map<Direction<3>, tnsr::i<DataVector, 3, Frame::Inertial>>
        unnormalized_normal_covectors{};
    tnsr::i<DataVector, 3, Frame::Inertial> unnormalized_covector{};
    const auto element_logical_to_grid_inv_jac =
        logical_to_grid_map.inv_jacobian(face_logical_coords);
    const auto grid_to_inertial_inv_jac = grid_to_inertial_map.inv_jacobian(
        logical_to_grid_map(face_logical_coords), time, functions_of_time);
    InverseJacobian<DataVector, 3, Frame::ElementLogical, Frame::Inertial>
        element_logical_to_inertial_inv_jac{};
    for (size_t logical_i = 0; logical_i < 3; ++logical_i) {
      for (size_t inertial_i = 0; inertial_i < 3; ++inertial_i) {
        element_logical_to_inertial_inv_jac.get(logical_i, inertial_i) =
            element_logical_to_grid_inv_jac.get(logical_i, 0) *
            grid_to_inertial_inv_jac.get(0, inertial_i);
        for (size_t grid_i = 1; grid_i < 3; ++grid_i) {
          element_logical_to_inertial_inv_jac.get(logical_i, inertial_i) +=
              element_logical_to_grid_inv_jac.get(logical_i, grid_i) *
              grid_to_inertial_inv_jac.get(grid_i, inertial_i);
        }
      }
    }
    for (size_t i = 0; i < 3; ++i) {
      unnormalized_covector.get(i) =
          element_logical_to_inertial_inv_jac.get(direction.dimension(), i);
    }
    unnormalized_normal_covectors[direction] = unnormalized_covector;
    Variables<tmpl::list<
        inverse_spatial_metric_tag,
        evolution::dg::Actions::detail::NormalVector<3>,
        evolution::dg::Actions::detail::OneOverNormalVectorMagnitude>>
        fields_on_face{face_mesh.number_of_grid_points()};
    fields_on_face.assign_subset(solution.variables(
        grid_to_inertial_map(logical_to_grid_map(face_logical_coords), time,
                             functions_of_time),
        time, tmpl::list<inverse_spatial_metric_tag>{}));
    normal_vectors[direction] = std::nullopt;
    evolution::dg::Actions::detail::
        unit_normal_vector_and_covector_and_magnitude<System>(
            make_not_null(&normal_vectors), make_not_null(&fields_on_face),
            direction, unnormalized_normal_covectors, grid_to_inertial_map);
  }

  auto box = db::create<db::AddSimpleTags<
      domain::Tags::Element<3>, domain::Tags::Mesh<3>,
      evolution::dg::subcell::Tags::Mesh<3>, typename System::variables_tag,
      Tags::TildeJ, typename System::spacetime_variables_tag,
      evolution::dg::subcell::Tags::OnSubcellFaces<
          typename System::flux_spacetime_variables_tag, 3>,
      evolution::dg::subcell::Tags::GhostDataForReconstruction<3>,
      fd::Tags::Reconstructor, evolution::Tags::BoundaryCorrection<System>,
      ::Tags::Time, domain::Tags::FunctionsOfTimeInitialize,
      domain::Tags::ElementMap<3, Frame::Grid>,
      domain::CoordinateMaps::Tags::CoordinateMap<3, Frame::Grid,
                                                  Frame::Inertial>,
      evolution::dg::subcell::Tags::Coordinates<3, Frame::ElementLogical>,
      evolution::dg::subcell::Tags::Coordinates<3, Frame::Inertial>,
      domain::Tags::MeshVelocity<3>,
      evolution::dg::Tags::NormalCovectorAndMagnitude<3>,
      ::Tags::AnalyticSolution<SolutionForTest>,
      evolution::dg::subcell::Tags::SubcellOptions<3>>>(
      element, dg_mesh, subcell_mesh, dg_evolved_vars, dg_tilde_j, dg_gr_vars,
      face_centered_gr_vars, ghost_data,
      std::unique_ptr<fd::Reconstructor>{
          std::make_unique<ReconstructionForTest>()},
      std::unique_ptr<BoundaryCorrections::BoundaryCorrection>{
          std::make_unique<BoundaryCorrectionForTest>()},
      time, clone_unique_ptrs(functions_of_time),
      ElementMap<3, Frame::Grid>{
          ElementId<3>{0},
          domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(
              domain::CoordinateMaps::Identity<3>{})},
      domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
          domain::CoordinateMaps::Identity<3>{}),
      subcell_logical_coords, subcell_inertial_coords,
      std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>{}, normal_vectors,
      solution,
      evolution::dg::subcell::SubcellOptions{
          4.0, 1_st, 1.0e-3, 1.0e-4, false,
          evolution::dg::subcell::fd::ReconstructionMethod::DimByDim, false,
          std::nullopt, ::fd::DerivativeOrder::Two, 1, 1, 1});

  // Compute the packaged data
  std::vector<DirectionalId<3>> mortars_to_reconstruct_to{};
  for (const auto& [direction, neighbors] : element.neighbors()) {
    mortars_to_reconstruct_to.emplace_back(
        DirectionalId<3>{direction, *neighbors.begin()});
  }
  const auto packaged_data =
      subcell::NeighborPackagedData::apply(box, mortars_to_reconstruct_to);

  // Now for each directions, check that the packaged_data agrees with
  // expected values
  BoundaryCorrectionForTest boundary_corr_for_test{};
  using dg_package_field_tags =
      typename BoundaryCorrectionForTest::dg_package_field_tags;
  using dg_package_data_temporary_tags =
      typename BoundaryCorrectionForTest::dg_package_data_temporary_tags;

  Variables<tmpl::append<
      tmpl::list<Tags::TildeJ>, evolved_vars_tags, fluxes_tags,
      tmpl::remove_duplicates<tmpl::push_back<
          dg_package_data_temporary_tags,
          gr::Tags::SpatialMetric<DataVector, 3>,
          gr::Tags::SqrtDetSpatialMetric<DataVector>,
          gr::Tags::InverseSpatialMetric<DataVector, 3, Frame::Inertial>,
          evolution::dg::Actions::detail::NormalVector<3>>>>>
      vars_on_mortar_face{0};
  Variables<dg_package_field_tags> expected_fd_packaged_data_on_mortar{0};

  for (const auto& mortar_id : mortars_to_reconstruct_to) {
    const Direction<3>& direction = mortar_id.direction();
    const size_t dim = direction.dimension();
    Index<3> extents = subcell_mesh.extents();

    ++extents[dim];
    const size_t num_mortar_face_pts =
        subcell_mesh.extents().slice_away(dim).product();

    vars_on_mortar_face.initialize(num_mortar_face_pts);

    // reconstruct TildeJ and evolved vars on the mortar
    auto recons_vars_on_mortar_face =
        vars_on_mortar_face.template reference_subset<
            ForceFree::fd::tags_list_for_reconstruction>();
    dynamic_cast<const ReconstructionForTest&>(reconstructor)
        .reconstruct_fd_neighbor(make_not_null(&recons_vars_on_mortar_face),
                                 volume_vars_subcell, subcell_tilde_j, element,
                                 ghost_data, subcell_mesh, direction);

    // retrieve face-centered GR vars and slice it on the mortar, then
    // compute fluxes with it
    tmpl::for_each<System::flux_spacetime_variables_tag::tags_list>(
        [&dim, &direction, &extents, &vars_on_mortar_face,
         &gr_vars_on_faces =
             db::get<evolution::dg::subcell::Tags::OnSubcellFaces<
                 typename System::flux_spacetime_variables_tag, 3>>(box)](
            auto tag_v) {
          using tag = tmpl::type_from<decltype(tag_v)>;
          data_on_slice(make_not_null(&get<tag>(vars_on_mortar_face)),
                        get<tag>(gsl::at(gr_vars_on_faces, dim)), extents, dim,
                        direction.side() == Side::Lower
                            ? 0
                            : extents[direction.dimension()] - 1);
        });
    subcell::compute_fluxes(make_not_null(&vars_on_mortar_face));

    // revert normal vector and project to DG grid
    auto normal_covector = get<evolution::dg::Tags::NormalCovector<3>>(
        *db::get<evolution::dg::Tags::NormalCovectorAndMagnitude<3>>(box).at(
            mortar_id.direction()));
    for (auto& t : normal_covector) {
      t *= -1.0;
    }
    const auto dg_normal_covector = normal_covector;
    for (size_t i = 0; i < 3; ++i) {
      normal_covector.get(i) = evolution::dg::subcell::fd::project(
          dg_normal_covector.get(i),
          dg_mesh.slice_away(mortar_id.direction().dimension()),
          subcell_mesh.extents().slice_away(mortar_id.direction().dimension()));
    }

    // Compute the expected packaged data
    expected_fd_packaged_data_on_mortar.initialize(num_mortar_face_pts);
    using dg_package_data_projected_tags =
        tmpl::append<evolved_vars_tags, fluxes_tags,
                     dg_package_data_temporary_tags>;
    evolution::dg::Actions::detail::dg_package_data<System>(
        make_not_null(&expected_fd_packaged_data_on_mortar),
        boundary_corr_for_test, vars_on_mortar_face, normal_covector,
        {std::nullopt}, box,
        typename BoundaryCorrectionForTest::dg_package_data_volume_tags{},
        dg_package_data_projected_tags{});

    // reconstruct from FD to DG grid for 2d
    auto expected_dg_packaged_data = evolution::dg::subcell::fd::reconstruct(
        expected_fd_packaged_data_on_mortar,
        dg_mesh.slice_away(mortar_id.direction().dimension()),
        subcell_mesh.extents().slice_away(mortar_id.direction().dimension()),
        evolution::dg::subcell::fd::ReconstructionMethod::DimByDim);

    const DataVector vector_to_check{expected_dg_packaged_data.data(),
                                     expected_dg_packaged_data.size()};

    CHECK_ITERABLE_APPROX(vector_to_check, packaged_data.at(mortar_id));
  }
}

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.ForceFree.Subcell.NeighborPackagedData",
    "[Unit][Evolution]") {
  MAKE_GENERATOR(gen);
  test_neighbor_packaged_data(make_not_null(&gen));
}

}  // namespace
}  // namespace ForceFree
