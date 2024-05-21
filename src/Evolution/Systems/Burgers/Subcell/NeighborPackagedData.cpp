// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Burgers/Subcell/NeighborPackagedData.hpp"

#include <cstddef>
#include <optional>
#include <type_traits>

#include "DataStructures/DataBox/Access.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/BoundaryCorrectionTags.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/NeighborReconstructedFaceSolution.tpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/OnSubcellFaces.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/PackageDataImpl.hpp"
#include "Evolution/DiscontinuousGalerkin/BoundaryData.hpp"
#include "Evolution/DiscontinuousGalerkin/NormalVectorTags.hpp"
#include "Evolution/Systems/Burgers/BoundaryCorrections/BoundaryCorrection.hpp"
#include "Evolution/Systems/Burgers/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/Burgers/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/Burgers/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/Burgers/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Burgers/Fluxes.hpp"
#include "Evolution/Systems/Burgers/Subcell/ComputeFluxes.hpp"
#include "Evolution/Systems/Burgers/System.hpp"
#include "Evolution/Systems/Burgers/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/TMPL.hpp"

namespace Burgers::subcell {
DirectionalIdMap<1, DataVector> NeighborPackagedData::apply(
    const db::Access& box,
    const std::vector<DirectionalId<1>>& mortars_to_reconstruct_to) {
  // The object to return
  DirectionalIdMap<1, DataVector> neighbor_package_data{};
  if (mortars_to_reconstruct_to.empty()) {
    return neighbor_package_data;
  }

  using evolved_vars_tags = typename System::variables_tag::tags_list;
  using fluxes_tags = typename Fluxes::return_tags;

  // subcell currently does not support moving mesh
  ASSERT(not db::get<domain::Tags::MeshVelocity<1>>(box).has_value(),
         "Haven't yet added support for moving mesh to DG-subcell. This "
         "should be easy to generalize, but we will want to consider "
         "storing the mesh velocity on the faces instead of "
         "re-slicing/projecting.");

  // Project volume variables from DG to subcell mesh
  const Mesh<1>& dg_mesh = db::get<domain::Tags::Mesh<1>>(box);
  const Mesh<1>& subcell_mesh =
      db::get<evolution::dg::subcell::Tags::Mesh<1>>(box);
  const auto volume_vars_subcell = evolution::dg::subcell::fd::project(
      db::get<typename System::variables_tag>(box), dg_mesh,
      subcell_mesh.extents());

  const auto& ghost_subcell_data =
      db::get<evolution::dg::subcell::Tags::GhostDataForReconstruction<1>>(box);

  const Burgers::fd::Reconstructor& recons =
      db::get<Burgers::fd::Tags::Reconstructor>(box);

  const auto& boundary_correction =
      db::get<evolution::Tags::BoundaryCorrection<System>>(box);
  using derived_boundary_corrections =
      typename std::decay_t<decltype(boundary_correction)>::creatable_classes;

  // perform reconstruction
  tmpl::for_each<derived_boundary_corrections>([&](auto derived_correction_v) {
    using derived_correction = tmpl::type_from<decltype(derived_correction_v)>;
    if (typeid(boundary_correction) == typeid(derived_correction)) {
      using dg_package_field_tags =
          typename derived_correction::dg_package_field_tags;
      using dg_package_data_argument_tags =
          tmpl::append<evolved_vars_tags, fluxes_tags>;

      const auto& element = db::get<domain::Tags::Element<1>>(box);

      // Variables to store packaged data
      Variables<dg_package_field_tags> packaged_data{};
      // Variables to be reconstructed on the shared interfaces
      Variables<dg_package_data_argument_tags> vars_on_face{};

      for (const auto& mortar_id : mortars_to_reconstruct_to) {
        // Note : 1D mortar has only one face point
        const size_t num_face_pts = 1;
        vars_on_face.initialize(num_face_pts);

        // Reconstruct field variables on faces
        call_with_dynamic_type<
            void, typename Burgers::fd::Reconstructor::creatable_classes>(
            &recons,
            [&element, &mortar_id, &ghost_subcell_data, &subcell_mesh,
             &vars_on_face, &volume_vars_subcell](const auto& reconstructor) {
              reconstructor->reconstruct_fd_neighbor(
                  make_not_null(&vars_on_face), volume_vars_subcell, element,
                  ghost_subcell_data, subcell_mesh, mortar_id.direction());
            });

        // Compute fluxes
        Burgers::subcell::compute_fluxes(make_not_null(&vars_on_face));

        tnsr::i<DataVector, 1, Frame::Inertial> normal_covector =
            get<evolution::dg::Tags::NormalCovector<1>>(
                *db::get<evolution::dg::Tags::NormalCovectorAndMagnitude<1>>(
                     box)
                     .at(mortar_id.direction()));
        for (auto& t : normal_covector) {
          t *= -1.0;
        }

        // Compute the packaged data
        packaged_data.initialize(num_face_pts);
        evolution::dg::Actions::detail::dg_package_data<System>(
            make_not_null(&packaged_data),
            dynamic_cast<const derived_correction&>(boundary_correction),
            vars_on_face, normal_covector, {std::nullopt}, box,
            typename derived_correction::dg_package_data_volume_tags{},
            dg_package_data_argument_tags{});

        // Make a view so we can use iterators with std::copy
        DataVector packaged_data_view{packaged_data.data(),
                                      packaged_data.size()};
        neighbor_package_data[mortar_id] = DataVector{packaged_data.size()};
        std::copy(packaged_data_view.begin(), packaged_data_view.end(),
                  neighbor_package_data[mortar_id].begin());
      }
    }
  });

  return neighbor_package_data;
}
}  // namespace Burgers::subcell

template void evolution::dg::subcell::neighbor_reconstructed_face_solution<
    1, Burgers::subcell::NeighborPackagedData>(
    gsl::not_null<db::Access*> box,
    gsl::not_null<std::pair<
        const TimeStepId, DirectionalIdMap<1, evolution::dg::BoundaryData<1>>>*>
        received_temporal_id_and_data);
