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
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/NeighborData.hpp"
#include "Evolution/DgSubcell/Tags/OnSubcellFaces.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/NormalCovectorAndMagnitude.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/PackageDataImpl.hpp"
#include "Evolution/Systems/Burgers/BoundaryCorrections/BoundaryCorrection.hpp"
#include "Evolution/Systems/Burgers/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/Burgers/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/Burgers/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Burgers/Subcell/ComputeFluxes.hpp"
#include "Evolution/Systems/Burgers/System.hpp"
#include "Evolution/Systems/Burgers/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Burgers::subcell {
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
      maximum_number_of_neighbors(1), std::pair<Direction<1>, ElementId<1>>,
      std::vector<double>, boost::hash<std::pair<Direction<1>, ElementId<1>>>>
  apply(const db::DataBox<DbTagsList>& box,
        const std::vector<std::pair<Direction<1>, ElementId<1>>>&
            mortars_to_reconstruct_to) noexcept {
    using evolved_vars_tag = typename System::variables_tag;
    using evolved_vars_tags = typename evolved_vars_tag::tags_list;
    using fluxes_tags = db::wrap_tags_in<::Tags::Flux, evolved_vars_tags,
                                         tmpl::size_t<1>, Frame::Inertial>;
    ASSERT(not db::get<domain::Tags::MeshVelocity<1>>(box).has_value(),
           "Haven't yet added support for moving mesh to DG-subcell. This "
           "should be easy to generalize, but we will want to consider "
           "storing the mesh velocity on the faces instead of "
           "re-slicing/projecting.");

    FixedHashMap<maximum_number_of_neighbors(1),
                 std::pair<Direction<1>, ElementId<1>>, std::vector<double>,
                 boost::hash<std::pair<Direction<1>, ElementId<1>>>>
        nhbr_package_data{};

    const auto& nhbr_subcell_data =
        db::get<evolution::dg::subcell::Tags::
                    NeighborDataForReconstructionAndRdmpTci<1>>(box);
    const Mesh<1>& subcell_mesh =
        db::get<evolution::dg::subcell::Tags::Mesh<1>>(box);

    Variables<tmpl::list<Burgers::Tags::U>> volume_vars(
        subcell_mesh.number_of_grid_points());
    get<Burgers::Tags::U>(volume_vars) =
        db::get<evolution::dg::subcell::Tags::Inactive<Burgers::Tags::U>>(box);

    const auto& recons = db::get<Burgers::fd::Tags::Reconstructor>(box);
    const auto& boundary_correction =
        db::get<evolution::Tags::BoundaryCorrection<Burgers::System>>(box);
    using derived_boundary_corrections =
        typename std::decay_t<decltype(boundary_correction)>::creatable_classes;
    tmpl::for_each<derived_boundary_corrections>(
        [&box, &boundary_correction, &mortars_to_reconstruct_to,
         &nhbr_package_data, &nhbr_subcell_data, &recons, &subcell_mesh,
         &volume_vars](auto derived_correction_v) noexcept {
          using DerivedCorrection =
              tmpl::type_from<decltype(derived_correction_v)>;
          if (typeid(boundary_correction) == typeid(DerivedCorrection)) {
            using dg_package_data_temporary_tags =
                typename DerivedCorrection::dg_package_data_temporary_tags;
            using dg_package_data_argument_tags =
                tmpl::append<evolved_vars_tags, fluxes_tags,
                             dg_package_data_temporary_tags>;

            const auto& element = db::get<domain::Tags::Element<1>>(box);

            Variables<dg_package_data_argument_tags> vars_on_face{1};
            for (const auto& mortar_id : mortars_to_reconstruct_to) {
              call_with_dynamic_type<
                  void, typename Burgers::fd::Reconstructor::creatable_classes>(
                  &recons, [&element, &mortar_id, &nhbr_subcell_data,
                            &subcell_mesh, &vars_on_face,
                            &volume_vars](const auto& reconstructor) noexcept {
                    reconstructor->reconstruct_fd_neighbor(
                        make_not_null(&vars_on_face), volume_vars, element,
                        nhbr_subcell_data, subcell_mesh, mortar_id.first);
                  });

              Burgers::subcell::compute_fluxes(make_not_null(&vars_on_face));

              tnsr::i<DataVector, 1, Frame::Inertial> normal_covector = get<
                  evolution::dg::Tags::NormalCovector<1>>(
                  *db::get<evolution::dg::Tags::NormalCovectorAndMagnitude<1>>(
                       box)
                       .at(mortar_id.first));
              for (auto& t : normal_covector) {
                t *= -1.0;
              }

              // Compute the packaged data
              Variables<typename DerivedCorrection::dg_package_field_tags>
                  packaged_data{1};
              using dg_package_data_projected_tags =
                  tmpl::append<evolved_vars_tags, fluxes_tags,
                               dg_package_data_temporary_tags>;
              evolution::dg::Actions::detail::dg_package_data<System>(
                  make_not_null(&packaged_data),
                  dynamic_cast<const DerivedCorrection&>(boundary_correction),
                  vars_on_face, normal_covector, {std::nullopt}, box,
                  typename DerivedCorrection::dg_package_data_volume_tags{},
                  dg_package_data_projected_tags{});

              nhbr_package_data[mortar_id] = std::vector<double>{
                  packaged_data.data(),
                  packaged_data.data() + packaged_data.size()};
            }
          }
        });

    return nhbr_package_data;
  }
};
}  // namespace Burgers::subcell
