// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ForceFree/Subcell/NeighborPackagedData.hpp"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/Access.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Slice.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Side.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/BoundaryCorrectionTags.hpp"
#include "Evolution/DgSubcell/NeighborReconstructedFaceSolution.tpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/Reconstruction.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/OnSubcellFaces.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/PackageDataImpl.hpp"
#include "Evolution/DiscontinuousGalerkin/BoundaryData.hpp"
#include "Evolution/DiscontinuousGalerkin/NormalVectorTags.hpp"
#include "Evolution/Systems/ForceFree/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/ForceFree/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/ForceFree/FiniteDifference/ReconstructWork.hpp"
#include "Evolution/Systems/ForceFree/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/ForceFree/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/ForceFree/Fluxes.hpp"
#include "Evolution/Systems/ForceFree/Subcell/ComputeFluxes.hpp"
#include "Evolution/Systems/ForceFree/System.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace ForceFree::subcell {
DirectionalIdMap<3, DataVector> NeighborPackagedData::apply(
    const db::Access& box,
    const std::vector<DirectionalId<3>>& mortars_to_reconstruct_to) {
  using evolved_vars_tags = typename System::variables_tag::tags_list;
  using fluxes_tags = typename Fluxes::return_tags;

  // Subcell with moving mesh is not supported in the ForceFree system yet.
  ASSERT(not db::get<domain::Tags::MeshVelocity<3>>(box).has_value(),
         "Haven't yet added support for moving mesh to DG-subcell. This "
         "should be easy to generalize, but we will want to consider "
         "storing the mesh velocity on the faces instead of "
         "re-slicing/projecting.");

  DirectionalIdMap<3, DataVector> neighbor_package_data{};
  if (mortars_to_reconstruct_to.empty()) {
    return neighbor_package_data;
  }

  // Project volume variables from DG to subcell mesh
  const Mesh<3>& dg_mesh = db::get<domain::Tags::Mesh<3>>(box);
  const Mesh<3>& subcell_mesh =
      db::get<evolution::dg::subcell::Tags::Mesh<3>>(box);
  const auto volume_vars_subcell = evolution::dg::subcell::fd::project(
      db::get<typename System::variables_tag>(box), dg_mesh,
      subcell_mesh.extents());

  // In addition to evolved vars, compute TildeJ on the subcell grid so that
  // it could be sent to neighbor elements doing FD
  DataVector buffer{
      Variables<tmpl::list<Tags::TildeJ>>::number_of_independent_components *
      dg_mesh.number_of_grid_points()};
  const auto volume_tildej_subcell = [&box, &buffer, &dg_mesh,
                                      &subcell_mesh]() {
    Variables<tmpl::list<Tags::TildeJ>> var{buffer.data(), buffer.size()};
    for (size_t i = 0; i < 3; ++i) {
      get<Tags::TildeJ>(var).get(i) = db::get<Tags::TildeJ>(box).get(i);
    }
    return get<Tags::TildeJ>(evolution::dg::subcell::fd::project(
        var, dg_mesh, subcell_mesh.extents()));
  }();

  const auto& ghost_subcell_data =
      db::get<evolution::dg::subcell::Tags::GhostDataForReconstruction<3>>(box);

  const ForceFree::fd::Reconstructor& recons =
      db::get<ForceFree::fd::Tags::Reconstructor>(box);

  const auto& boundary_correction =
      db::get<evolution::Tags::BoundaryCorrection<System>>(box);
  using derived_boundary_corrections =
      typename std::decay_t<decltype(boundary_correction)>::creatable_classes;

  const auto& subcell_options =
      db::get<evolution::dg::subcell::Tags::SubcellOptions<3>>(box);

  tmpl::for_each<derived_boundary_corrections>([&box, &boundary_correction,
                                                &dg_mesh,
                                                &mortars_to_reconstruct_to,
                                                &neighbor_package_data,
                                                &ghost_subcell_data, &recons,
                                                &subcell_mesh, &subcell_options,
                                                &volume_vars_subcell,
                                                &volume_tildej_subcell](
                                                   auto derived_correction_v) {
    using DerivedCorrection = tmpl::type_from<decltype(derived_correction_v)>;

    using dg_package_field_tags =
        typename DerivedCorrection::dg_package_field_tags;
    using dg_package_data_temporary_tags =
        typename DerivedCorrection::dg_package_data_temporary_tags;

    if (typeid(boundary_correction) == typeid(DerivedCorrection)) {
      Variables<dg_package_field_tags> packaged_data{0};

      // Comprehensive tags list for computing fluxes and dg_package_data
      using dg_package_data_argument_tags = tmpl::append<
          evolved_vars_tags, fluxes_tags,
          tmpl::remove_duplicates<tmpl::push_back<
              dg_package_data_temporary_tags,
              gr::Tags::SpatialMetric<DataVector, 3>,
              gr::Tags::SqrtDetSpatialMetric<DataVector>,
              gr::Tags::InverseSpatialMetric<DataVector, 3, Frame::Inertial>,
              evolution::dg::Actions::detail::NormalVector<3>>>>;

      // We allocate a Variables object with extra buffer for storing TildeJ.
      // TildeJ is not an argument of dg_package_data() function, but it needs
      // to be reconstructed for computing fluxes on faces.
      Variables<tmpl::append<tmpl::list<ForceFree::Tags::TildeJ>,
                             dg_package_data_argument_tags>>
          vars_on_face_with_tildej_buffer{0};

      for (const auto& mortar_id : mortars_to_reconstruct_to) {
        const Direction<3>& direction = mortar_id.direction();

        // Switch to face-centered points
        Index<3> extents = subcell_mesh.extents();
        ++extents[direction.dimension()];
        const size_t num_face_pts =
            subcell_mesh.extents().slice_away(direction.dimension()).product();
        vars_on_face_with_tildej_buffer.initialize(num_face_pts);

        // create a `view` of the face vars excluding the TildeJ tag, since
        // boundary correction takes the flux F^i(TildeQ) as argument but not
        // TildeJ itself.
        auto vars_on_face =
            vars_on_face_with_tildej_buffer
                .template reference_subset<dg_package_data_argument_tags>();
        auto reconstructed_vars_on_face =
            vars_on_face_with_tildej_buffer.template reference_subset<
                ForceFree::fd::tags_list_for_reconstruction>();

        // Copy spacetime vars over from volume.
        using spacetime_vars_to_copy = tmpl::list<
            gr::Tags::Lapse<DataVector>,
            gr::Tags::Shift<DataVector, 3, Frame::Inertial>,
            gr::Tags::SpatialMetric<DataVector, 3>,
            gr::Tags::SqrtDetSpatialMetric<DataVector>,
            gr::Tags::InverseSpatialMetric<DataVector, 3, Frame::Inertial>>;
        tmpl::for_each<spacetime_vars_to_copy>(
            [&direction, &extents, &vars_on_face,
             &spacetime_vars_on_faces =
                 db::get<evolution::dg::subcell::Tags::OnSubcellFaces<
                     typename System::flux_spacetime_variables_tag, 3>>(box)](
                auto tag_v) {
              using tag = tmpl::type_from<decltype(tag_v)>;
              data_on_slice(make_not_null(&get<tag>(vars_on_face)),
                            get<tag>(gsl::at(spacetime_vars_on_faces,
                                             direction.dimension())),
                            extents, direction.dimension(),
                            direction.side() == Side::Lower
                                ? 0
                                : extents[direction.dimension()] - 1);
            });

        // Perform FD reconstruction of variables on cell interfaces. Note
        // that we are using the vars object with TildeJ buffer so that
        // TildeJ can be reconstructed in this step.
        call_with_dynamic_type<
            void, typename ForceFree::fd::Reconstructor::creatable_classes>(
            &recons, [&element = db::get<domain::Tags::Element<3>>(box),
                      &mortar_id, &ghost_subcell_data, &subcell_mesh,
                      &reconstructed_vars_on_face, &volume_vars_subcell,
                      &volume_tildej_subcell](const auto& reconstructor) {
              reconstructor->reconstruct_fd_neighbor(
                  make_not_null(&reconstructed_vars_on_face),
                  volume_vars_subcell, volume_tildej_subcell, element,
                  ghost_subcell_data, subcell_mesh, mortar_id.direction());
            });
        ForceFree::subcell::compute_fluxes(
            make_not_null(&vars_on_face_with_tildej_buffer));

        // Note: since the spacetime isn't dynamical we don't need to
        // worry about different normal vectors on the different sides
        // of the element. If the spacetime is dynamical, then the normal
        // covector needs to be sent, or at least the inverse spatial metric
        // from the neighbor so that the normal covector can be computed
        // correctly.
        tnsr::i<DataVector, 3, Frame::Inertial> normal_covector =
            get<evolution::dg::Tags::NormalCovector<3>>(
                *db::get<evolution::dg::Tags::NormalCovectorAndMagnitude<3>>(
                     box)
                     .at(mortar_id.direction()));
        for (auto& t : normal_covector) {
          t *= -1.0;
        }

        // Note: Only need to do the projection in 2d and 3d, but GRFFE is
        // always 3d currently.
        const auto dg_normal_covector = normal_covector;
        for (size_t i = 0; i < 3; ++i) {
          normal_covector.get(i) = evolution::dg::subcell::fd::project(
              dg_normal_covector.get(i),
              dg_mesh.slice_away(mortar_id.direction().dimension()),
              subcell_mesh.extents().slice_away(
                  mortar_id.direction().dimension()));
        }

        // Compute the packaged data
        packaged_data.initialize(num_face_pts);
        using dg_package_data_projected_tags =
            tmpl::append<evolved_vars_tags, fluxes_tags,
                         dg_package_data_temporary_tags>;
        evolution::dg::Actions::detail::dg_package_data<System>(
            make_not_null(&packaged_data),
            dynamic_cast<const DerivedCorrection&>(boundary_correction),
            vars_on_face, normal_covector, {std::nullopt}, box,
            typename DerivedCorrection::dg_package_data_volume_tags{},
            dg_package_data_projected_tags{});

        // Reconstruct the DG solution. Really we should be solving the
        // boundary correction and then reconstructing, but away from a shock
        // this doesn't matter.
        auto dg_packaged_data = evolution::dg::subcell::fd::reconstruct(
            packaged_data,
            dg_mesh.slice_away(mortar_id.direction().dimension()),
            subcell_mesh.extents().slice_away(
                mortar_id.direction().dimension()),
            subcell_options.reconstruction_method());

        // Make a view so we can use iterators with std::copy
        DataVector dg_packaged_data_view{dg_packaged_data.data(),
                                         dg_packaged_data.size()};
        neighbor_package_data[mortar_id] = DataVector{dg_packaged_data.size()};
        std::copy(dg_packaged_data_view.begin(), dg_packaged_data_view.end(),
                  neighbor_package_data[mortar_id].begin());
      }
    }
  });

  return neighbor_package_data;
}
}  // namespace ForceFree::subcell

template void evolution::dg::subcell::neighbor_reconstructed_face_solution<
    3, ForceFree::subcell::NeighborPackagedData>(
    gsl::not_null<db::Access*> box,
    gsl::not_null<std::pair<
        const TimeStepId, DirectionalIdMap<3, evolution::dg::BoundaryData<3>>>*>
        received_temporal_id_and_data);
