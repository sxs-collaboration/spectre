// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/NeighborPackagedData.hpp"

#include <cstddef>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/Access.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/BoundaryCorrectionTags.hpp"
#include "Evolution/DgSubcell/NeighborReconstructedFaceSolution.tpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/Reconstruction.hpp"
#include "Evolution/DgSubcell/ReconstructionMethod.hpp"
#include "Evolution/DgSubcell/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/OnSubcellFaces.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/NormalCovectorAndMagnitude.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/PackageDataImpl.hpp"
#include "Evolution/DiscontinuousGalerkin/BoundaryData.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/BoundaryCorrection.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/ReconstructWork.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Tag.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/ComputeFluxes.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::ValenciaDivClean::subcell {
DirectionalIdMap<3, DataVector> NeighborPackagedData::apply(
    const db::Access& box,
    const std::vector<DirectionalId<3>>& mortars_to_reconstruct_to) {
  using evolved_vars_tag = typename System::variables_tag;
  using evolved_vars_tags = typename evolved_vars_tag::tags_list;
  using fluxes_tags = db::wrap_tags_in<::Tags::Flux, evolved_vars_tags,
                                       tmpl::size_t<3>, Frame::Inertial>;

  ASSERT(not db::get<domain::Tags::MeshVelocity<3>>(box).has_value(),
         "Haven't yet added support for moving mesh to DG-subcell. This "
         "should be easy to generalize, but we will want to consider "
         "storing the mesh velocity on the faces instead of "
         "re-slicing/projecting.");

  DirectionalIdMap<3, DataVector> neighbor_package_data{};
  if (mortars_to_reconstruct_to.empty()) {
    return neighbor_package_data;
  }

  const auto& ghost_subcell_data =
      db::get<evolution::dg::subcell::Tags::GhostDataForReconstruction<3>>(box);
  const Mesh<3>& subcell_mesh =
      db::get<evolution::dg::subcell::Tags::Mesh<3>>(box);
  const Mesh<3>& dg_mesh = db::get<domain::Tags::Mesh<3>>(box);
  const auto& subcell_options =
      db::get<evolution::dg::subcell::Tags::SubcellOptions<3>>(box);

  // Note: we need to compare if projecting the entire mesh or only ghost
  // zones needed is faster. This probably depends on the number of neighbors
  // we have doing FD.
  const auto volume_prims = evolution::dg::subcell::fd::project(
      db::get<typename System::primitive_variables_tag>(box), dg_mesh,
      subcell_mesh.extents());

  const auto& recons =
      db::get<grmhd::ValenciaDivClean::fd::Tags::Reconstructor>(box);
  const auto& boundary_correction =
      db::get<evolution::Tags::BoundaryCorrection<System>>(box);
  using derived_boundary_corrections =
      typename std::decay_t<decltype(boundary_correction)>::creatable_classes;
  tmpl::for_each<derived_boundary_corrections>([&box, &boundary_correction,
                                                &dg_mesh,
                                                &mortars_to_reconstruct_to,
                                                &neighbor_package_data,
                                                &ghost_subcell_data, &recons,
                                                &subcell_mesh, &subcell_options,
                                                &volume_prims](
                                                   auto derived_correction_v) {
    using DerivedCorrection = tmpl::type_from<decltype(derived_correction_v)>;
    if (typeid(boundary_correction) == typeid(DerivedCorrection)) {
      using dg_package_data_temporary_tags =
          typename DerivedCorrection::dg_package_data_temporary_tags;
      using dg_package_data_argument_tags = fd::tags_list_for_reconstruct;

      const auto& element = db::get<domain::Tags::Element<3>>(box);
      const auto& eos = get<hydro::Tags::GrmhdEquationOfState>(box);

      using dg_package_field_tags =
          typename DerivedCorrection::dg_package_field_tags;
      Variables<dg_package_data_argument_tags> vars_on_face{0};
      Variables<dg_package_field_tags> packaged_data{0};
      for (const auto& mortar_id : mortars_to_reconstruct_to) {
        const Direction<3>& direction = mortar_id.direction();

        const Mesh<2> dg_face_mesh = dg_mesh.slice_away(direction.dimension());
        Index<3> extents = subcell_mesh.extents();
        // Switch to face-centered instead of cell-centered points on the FD.
        // There are num_cell_centered+1 face-centered points.
        ++extents[direction.dimension()];
        const Index<2> subcell_face_extents =
            extents.slice_away(direction.dimension());

        // Computed prims and cons on face via reconstruction
        const size_t num_face_pts =
            subcell_mesh.extents().slice_away(direction.dimension()).product();
        vars_on_face.initialize(num_face_pts);
        // Copy spacetime vars over from volume.
        using spacetime_vars_to_copy =
            tmpl::list<gr::Tags::Lapse<DataVector>,
                       gr::Tags::Shift<DataVector, 3>,
                       gr::Tags::SpatialMetric<DataVector, 3>,
                       gr::Tags::SqrtDetSpatialMetric<DataVector>,
                       gr::Tags::InverseSpatialMetric<DataVector, 3>>;
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

        call_with_dynamic_type<void, typename grmhd::ValenciaDivClean::fd::
                                         Reconstructor::creatable_classes>(
            &recons,
            [&element, &eos, &mortar_id, &ghost_subcell_data, &subcell_mesh,
             &vars_on_face, &volume_prims](const auto& reconstructor) {
              reconstructor->reconstruct_fd_neighbor(
                  make_not_null(&vars_on_face), volume_prims, eos, element,
                  ghost_subcell_data, subcell_mesh, mortar_id.direction());
            });

        // Get the mesh velocity if needed
        const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>
            mesh_velocity_dg = db::get<domain::Tags::MeshVelocity<3>>(box);
        std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>
            mesh_velocity_on_subcell_face = {};
        if (mesh_velocity_dg.has_value()) {
          // Slice data on current face
          tnsr::I<DataVector, 3, Frame::Inertial> mesh_velocity_on_dg_face =
              data_on_slice(mesh_velocity_dg.value(), dg_mesh.extents(),
                            direction.dimension(),
                            direction.side() == Side::Lower
                                ? 0
                                : (dg_mesh.extents(direction.dimension()) - 1));

          mesh_velocity_on_subcell_face =
              tnsr::I<DataVector, 3, Frame::Inertial>{num_face_pts};

          for (size_t i = 0; i < 3; i++) {
            evolution::dg::subcell::fd::project(
                make_not_null(&mesh_velocity_on_subcell_face.value().get(i)),
                mesh_velocity_on_dg_face.get(i), dg_face_mesh,
                subcell_face_extents);
          }
        }

        grmhd::ValenciaDivClean::subcell::compute_fluxes(
            make_not_null(&vars_on_face));

        if (mesh_velocity_on_subcell_face.has_value()) {
          tmpl::for_each<evolved_vars_tags>(
              [&vars_on_face, &mesh_velocity_on_subcell_face](auto tag_v) {
                using tag = tmpl::type_from<decltype(tag_v)>;
                using flux_tag =
                    ::Tags::Flux<tag, tmpl::size_t<3>, Frame::Inertial>;
                using FluxTensor = typename flux_tag::type;
                const auto& var = get<tag>(vars_on_face);
                auto& flux = get<flux_tag>(vars_on_face);
                for (size_t storage_index = 0; storage_index < var.size();
                     ++storage_index) {
                  const auto tensor_index = var.get_tensor_index(storage_index);
                  for (size_t j = 0; j < 3; j++) {
                    const auto flux_storage_index =
                        FluxTensor::get_storage_index(prepend(tensor_index, j));
                    flux[flux_storage_index] -=
                        mesh_velocity_on_subcell_face.value().get(j) *
                        var[storage_index];
                  }
                }
              });
        }

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
        // Note: Only need to do the projection in 2d and 3d, but GRMHD is
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
        using dg_package_data_projected_tags = tmpl::append<
            evolved_vars_tags, fluxes_tags, dg_package_data_temporary_tags,
            typename DerivedCorrection::dg_package_data_primitive_tags>;
        evolution::dg::Actions::detail::dg_package_data<System>(
            make_not_null(&packaged_data),
            dynamic_cast<const DerivedCorrection&>(boundary_correction),
            vars_on_face, normal_covector, mesh_velocity_on_subcell_face, box,
            typename DerivedCorrection::dg_package_data_volume_tags{},
            dg_package_data_projected_tags{});

        // Reconstruct the DG solution.
        // Really we should be solving the boundary correction and
        // then reconstructing, but away from a shock this doesn't
        // matter.
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
}  // namespace grmhd::ValenciaDivClean::subcell

template void evolution::dg::subcell::neighbor_reconstructed_face_solution<
    3, grmhd::ValenciaDivClean::subcell::NeighborPackagedData>(
    gsl::not_null<db::Access*> box,
    gsl::not_null<std::pair<
        const TimeStepId, DirectionalIdMap<3, evolution::dg::BoundaryData<3>>>*>
        received_temporal_id_and_data);
