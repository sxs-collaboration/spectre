// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/functional/hash.hpp>
#include <cstddef>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/BoundaryCorrectionTags.hpp"
#include "Evolution/DgSubcell/NeighborData.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/Reconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/NeighborData.hpp"
#include "Evolution/DgSubcell/Tags/OnSubcellFaces.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/NormalCovectorAndMagnitude.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/PackageDataImpl.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/BoundaryCorrection.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Tag.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/ComputeFluxes.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/FakeVirtual.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::ValenciaDivClean::subcell {
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
struct NeighborPackagedData {
  template <typename DbTagsList>
  static FixedHashMap<
      maximum_number_of_neighbors(3), std::pair<Direction<3>, ElementId<3>>,
      std::vector<double>, boost::hash<std::pair<Direction<3>, ElementId<3>>>>
  apply(const db::DataBox<DbTagsList>& box,
        const std::vector<std::pair<Direction<3>, ElementId<3>>>&
            mortars_to_reconstruct_to) noexcept {
    using evolved_vars_tag = typename System::variables_tag;
    using evolved_vars_tags = typename evolved_vars_tag::tags_list;
    using prim_tags = typename System::primitive_variables_tag::tags_list;
    using recons_prim_tags = tmpl::push_back<
        prim_tags,
        hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>>;
    using fluxes_tags = db::wrap_tags_in<::Tags::Flux, evolved_vars_tags,
                                         tmpl::size_t<3>, Frame::Inertial>;

    ASSERT(not db::get<domain::Tags::MeshVelocity<3>>(box).has_value(),
           "Haven't yet added support for moving mesh to DG-subcell. This "
           "should be easy to generalize, but we will want to consider "
           "storing the mesh velocity on the faces instead of "
           "re-slicing/projecting.");

    FixedHashMap<maximum_number_of_neighbors(3),
                 std::pair<Direction<3>, ElementId<3>>, std::vector<double>,
                 boost::hash<std::pair<Direction<3>, ElementId<3>>>>
        nhbr_package_data{};

    const auto& nhbr_subcell_data =
        db::get<evolution::dg::subcell::Tags::
                    NeighborDataForReconstructionAndRdmpTci<3>>(box);
    const Mesh<3>& subcell_mesh =
        db::get<evolution::dg::subcell::Tags::Mesh<3>>(box);
    const Mesh<3>& dg_mesh = db::get<domain::Tags::Mesh<3>>(box);

    const auto volume_prims = evolution::dg::subcell::fd::project(
        db::get<typename System::primitive_variables_tag>(box), dg_mesh,
        subcell_mesh.extents());

    const auto& recons =
        db::get<grmhd::ValenciaDivClean::fd::Tags::Reconstructor>(box);
    const auto& boundary_correction =
        db::get<evolution::Tags::BoundaryCorrection<System>>(box);
    using derived_boundary_corrections =
        typename std::decay_t<decltype(boundary_correction)>::creatable_classes;
    tmpl::for_each<
        derived_boundary_corrections>([&box, &boundary_correction, &dg_mesh,
                                       &mortars_to_reconstruct_to,
                                       &nhbr_package_data, &nhbr_subcell_data,
                                       &recons, &subcell_mesh, &volume_prims](
                                          auto derived_correction_v) noexcept {
      using DerivedCorrection = tmpl::type_from<decltype(derived_correction_v)>;
      if (typeid(boundary_correction) == typeid(DerivedCorrection)) {
        using dg_package_data_temporary_tags =
            typename DerivedCorrection::dg_package_data_temporary_tags;
        using dg_package_data_argument_tags = tmpl::append<
            evolved_vars_tags, recons_prim_tags, fluxes_tags,
            tmpl::remove_duplicates<tmpl::push_back<
                dg_package_data_temporary_tags, gr::Tags::SpatialMetric<3>,
                gr::Tags::SqrtDetSpatialMetric<DataVector>,
                gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataVector>,
                evolution::dg::Actions::detail::NormalVector<3>>>>;

        const auto& element = db::get<domain::Tags::Element<3>>(box);
        const auto& eos = get<hydro::Tags::EquationOfStateBase>(box);

        using dg_package_field_tags =
            typename DerivedCorrection::dg_package_field_tags;
        Variables<dg_package_data_argument_tags> vars_on_face{0};
        Variables<dg_package_field_tags> packaged_data{0};
        for (const auto& mortar_id : mortars_to_reconstruct_to) {
          const Direction<3>& direction = mortar_id.first;

          Index<3> extents = subcell_mesh.extents();
          // Switch to face-centered instead of cell-centered points on the FD.
          // There are num_cell_centered+1 face-centered points.
          ++extents[direction.dimension()];

          // Computed prims and cons on face via reconstruction
          const size_t num_face_pts = subcell_mesh.extents()
                                          .slice_away(direction.dimension())
                                          .product();
          vars_on_face.initialize(num_face_pts);
          // Copy spacetime vars over from volume.
          using spacetime_vars_to_copy = tmpl::list<
              gr::Tags::Lapse<DataVector>,
              gr::Tags::Shift<3, Frame::Inertial, DataVector>,
              gr::Tags::SpatialMetric<3>,
              gr::Tags::SqrtDetSpatialMetric<DataVector>,
              gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataVector>>;
          tmpl::for_each<spacetime_vars_to_copy>(
              [&direction, &extents, &vars_on_face,
               &spacetime_vars_on_faces =
                   db::get<evolution::dg::subcell::Tags::OnSubcellFaces<
                       typename System::flux_spacetime_variables_tag, 3>>(box)](
                  auto tag_v) noexcept {
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
              &recons, [&element, &eos, &mortar_id, &nhbr_subcell_data,
                        &subcell_mesh, &vars_on_face,
                        &volume_prims](const auto& reconstructor) noexcept {
                reconstructor->reconstruct_fd_neighbor(
                    make_not_null(&vars_on_face), volume_prims, eos, element,
                    nhbr_subcell_data, subcell_mesh, mortar_id.first);
              });

          grmhd::ValenciaDivClean::subcell::compute_fluxes(
              make_not_null(&vars_on_face));

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
                       .at(mortar_id.first));
          for (auto& t : normal_covector) {
            t *= -1.0;
          }
          // Note: Only need to do the projection in 2d and 3d, but GRMHD is
          // always 3d currently.
          const auto dg_normal_covector = normal_covector;
          for (size_t i = 0; i < 3; ++i) {
            normal_covector.get(i) = evolution::dg::subcell::fd::project(
                dg_normal_covector.get(i),
                dg_mesh.slice_away(mortar_id.first.dimension()),
                subcell_mesh.extents().slice_away(mortar_id.first.dimension()));
          }

          // Compute the packaged data
          packaged_data.initialize(num_face_pts);
          using dg_package_data_projected_tags = tmpl::append<
              evolved_vars_tags, fluxes_tags, dg_package_data_temporary_tags,
              typename DerivedCorrection::dg_package_data_primitive_tags>;
          evolution::dg::Actions::detail::dg_package_data<System>(
              make_not_null(&packaged_data),
              dynamic_cast<const DerivedCorrection&>(boundary_correction),
              vars_on_face, normal_covector, {std::nullopt}, box,
              typename DerivedCorrection::dg_package_data_volume_tags{},
              dg_package_data_projected_tags{});

          // Reconstruct the DG solution.
          // Really we should be solving the boundary correction and
          // then reconstructing, but away from a shock this doesn't
          // matter.
          auto dg_packaged_data = evolution::dg::subcell::fd::reconstruct(
              packaged_data, dg_mesh.slice_away(mortar_id.first.dimension()),
              subcell_mesh.extents().slice_away(mortar_id.first.dimension()));
          nhbr_package_data[mortar_id] = std::vector<double>{
              dg_packaged_data.data(),
              dg_packaged_data.data() + dg_packaged_data.size()};
        }
      }
    });

    return nhbr_package_data;
  }
};
}  // namespace grmhd::ValenciaDivClean::subcell
