// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/functional/hash.hpp>
#include <cstddef>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Slice.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/BoundaryCorrectionTags.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/Reconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/NeighborData.hpp"
#include "Evolution/DgSubcell/Tags/OnSubcellFaces.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/PackageDataImpl.hpp"
#include "Evolution/DiscontinuousGalerkin/NormalVectorTags.hpp"
#include "Evolution/Systems/ScalarAdvection/BoundaryCorrections/BoundaryCorrection.hpp"
#include "Evolution/Systems/ScalarAdvection/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/ScalarAdvection/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/ScalarAdvection/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/ScalarAdvection/Fluxes.hpp"
#include "Evolution/Systems/ScalarAdvection/Subcell/ComputeFluxes.hpp"
#include "Evolution/Systems/ScalarAdvection/Subcell/VelocityAtFace.hpp"
#include "Evolution/Systems/ScalarAdvection/System.hpp"
#include "Evolution/Systems/ScalarAdvection/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/FakeVirtual.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarAdvection::subcell {
/*!
 * \brief On elements using DG, reconstructs the interface data from a
 * neighboring element doing subcell.
 *
 * The neighbor's packaged data needed by the boundary correction is computed
 * and returned so that it can be used for solving the Riemann problem on the
 * interfaces.
 *
 * Note that for strict conservation the Riemann solve should be done on the
 * subcells, with the correction being projected back to the DG interface.
 * However, in practice such strict conservation doesn't seem to be necessary
 * and can be explained by that we only need strict conservation at shocks, and
 * if one element is doing DG, then we aren't at a shock.
 */
template <size_t Dim>
struct NeighborPackagedData {
  template <typename DbTagsList>
  static FixedHashMap<maximum_number_of_neighbors(Dim),
                      std::pair<Direction<Dim>, ElementId<Dim>>,
                      std::vector<double>,
                      boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>
  apply(const db::DataBox<DbTagsList>& box,
        const std::vector<std::pair<Direction<Dim>, ElementId<Dim>>>&
            mortars_to_reconstruct_to) {
    // The object to return
    FixedHashMap<maximum_number_of_neighbors(Dim),
                 std::pair<Direction<Dim>, ElementId<Dim>>, std::vector<double>,
                 boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>
        neighbor_package_data{};

    using evolved_vars_tags = typename System<Dim>::variables_tag::tags_list;
    using fluxes_tags = typename Fluxes<Dim>::return_tags;

    // subcell currently does not support moving mesh
    ASSERT(not db::get<domain::Tags::MeshVelocity<Dim>>(box).has_value(),
           "Haven't yet added support for moving mesh to DG-subcell. This "
           "should be easy to generalize, but we will want to consider "
           "storing the mesh velocity on the faces instead of "
           "re-slicing/projecting.");

    // Project volume variables from DG to subcell mesh
    const Mesh<Dim>& dg_mesh = db::get<domain::Tags::Mesh<Dim>>(box);
    const Mesh<Dim>& subcell_mesh =
        db::get<evolution::dg::subcell::Tags::Mesh<Dim>>(box);
    const auto volume_vars_subcell = evolution::dg::subcell::fd::project(
        db::get<typename System<Dim>::variables_tag>(box), dg_mesh,
        subcell_mesh.extents());

    const auto& neighbor_subcell_data =
        db::get<evolution::dg::subcell::Tags::
                    NeighborDataForReconstructionAndRdmpTci<Dim>>(box);

    const ScalarAdvection::fd::Reconstructor<Dim>& recons =
        db::get<ScalarAdvection::fd::Tags::Reconstructor<Dim>>(box);

    const auto& boundary_correction =
        db::get<evolution::Tags::BoundaryCorrection<System<Dim>>>(box);
    using derived_boundary_corrections =
        typename std::decay_t<decltype(boundary_correction)>::creatable_classes;

    // perform reconstruction
    tmpl::for_each<derived_boundary_corrections>([&](auto
                                                         derived_correction_v) {
      using derived_correction =
          tmpl::type_from<decltype(derived_correction_v)>;
      if (typeid(boundary_correction) == typeid(derived_correction)) {
        using dg_package_field_tags =
            typename derived_correction::dg_package_field_tags;
        using dg_package_data_temporary_tags =
            typename derived_correction::dg_package_data_temporary_tags;
        using dg_package_data_argument_tags =
            tmpl::append<evolved_vars_tags, fluxes_tags,
                         dg_package_data_temporary_tags>;

        const auto& element = db::get<domain::Tags::Element<Dim>>(box);

        // Variables to store packaged data
        Variables<dg_package_field_tags> packaged_data{};
        // Variables to be reconstructed on the shared interfaces
        Variables<dg_package_data_argument_tags> vars_on_face{};

        for (const auto& mortar_id : mortars_to_reconstruct_to) {
          const Direction<Dim>& direction = mortar_id.first;
          Index<Dim> extents = subcell_mesh.extents();

          size_t num_face_pts{std::numeric_limits<size_t>::signaling_NaN()};

          if constexpr (Dim == 1) {
            // Note : 1D mortar has only one face point
            num_face_pts = 1;
            vars_on_face.initialize(num_face_pts);
          } else {
            // Switch to face-centered instead of cell-centered points on the
            // FD. There are num_cell_centered+1 face-centered points.
            ++extents[direction.dimension()];
            num_face_pts = subcell_mesh.extents()
                               .slice_away(direction.dimension())
                               .product();
            vars_on_face.initialize(num_face_pts);
          }

          using velocity_field = ScalarAdvection::Tags::VelocityField<Dim>;
          const auto& velocity_on_face =
              db::get<evolution::dg::subcell::Tags::OnSubcellFaces<
                  velocity_field, Dim>>(box);
          data_on_slice(make_not_null(&get<velocity_field>(vars_on_face)),
                        gsl::at(velocity_on_face, direction.dimension()),
                        extents, direction.dimension(),
                        direction.side() == Side::Lower
                            ? 0
                            : extents[direction.dimension()] - 1);

          // Reconstruct field variables on faces
          call_with_dynamic_type<void,
                                 typename ScalarAdvection::fd::Reconstructor<
                                     Dim>::creatable_classes>(
              &recons,
              [&element, &mortar_id, &neighbor_subcell_data, &subcell_mesh,
               &vars_on_face, &volume_vars_subcell](const auto& reconstructor) {
                reconstructor->reconstruct_fd_neighbor(
                    make_not_null(&vars_on_face), volume_vars_subcell, element,
                    neighbor_subcell_data, subcell_mesh, mortar_id.first);
              });

          // Compute fluxes
          ScalarAdvection::subcell::compute_fluxes<Dim>(
              make_not_null(&vars_on_face));

          tnsr::i<DataVector, Dim, Frame::Inertial> normal_covector = get<
              evolution::dg::Tags::NormalCovector<Dim>>(
              *db::get<evolution::dg::Tags::NormalCovectorAndMagnitude<Dim>>(
                   box)
                   .at(mortar_id.first));
          for (auto& t : normal_covector) {
            t *= -1.0;
          }

          // Note: need to do the projection in 2D
          if constexpr (Dim > 1) {
            const auto dg_normal_covector = normal_covector;
            for (size_t i = 0; i < Dim; ++i) {
              normal_covector.get(i) = evolution::dg::subcell::fd::project(
                  dg_normal_covector.get(i),
                  dg_mesh.slice_away(mortar_id.first.dimension()),
                  subcell_mesh.extents().slice_away(
                      mortar_id.first.dimension()));
            }
          }

          // Compute the packaged data
          packaged_data.initialize(num_face_pts);
          evolution::dg::Actions::detail::dg_package_data<System<Dim>>(
              make_not_null(&packaged_data),
              dynamic_cast<const derived_correction&>(boundary_correction),
              vars_on_face, normal_covector, {std::nullopt}, box,
              typename derived_correction::dg_package_data_volume_tags{},
              dg_package_data_argument_tags{});

          neighbor_package_data[mortar_id] =
              std::vector<double>{packaged_data.data(),
                                  packaged_data.data() + packaged_data.size()};

          // Note : need to reconstruct from FD to DG grid in 2D
          if constexpr (Dim > 1) {
            const auto dg_packaged_data =
                evolution::dg::subcell::fd::reconstruct(
                    packaged_data,
                    dg_mesh.slice_away(mortar_id.first.dimension()),
                    subcell_mesh.extents().slice_away(
                        mortar_id.first.dimension()));
            neighbor_package_data[mortar_id] = std::vector<double>{
                dg_packaged_data.data(),
                dg_packaged_data.data() + dg_packaged_data.size()};
          }
        }
      }
    });

    return neighbor_package_data;
  }
};
}  // namespace ScalarAdvection::subcell
