// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/functional/hash.hpp>  // IWYU pragma: keep
#include <cstddef>
#include <map>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"  // IWYU pragma: keep
#include "DataStructures/VariablesHelpers.hpp"
#include "Domain/Direction.hpp"  // IWYU pragma: keep
#include "Domain/ElementId.hpp"  // IWYU pragma: keep
#include "Domain/Tags.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace dg {
/*!
 * \ingroup DiscontinuousGalerkinGroup
 * \brief Types related to flux communication.
 *
 * \tparam Dim The spatial dimensions of the system.
 * \tparam TemporalIdTag The databox tag of the type that identifies a step in
 * the algorithm, e.g. Tags::TimeId.
 * \tparam NumericalFluxTag The cache tag for the type responsible for computing
 * the numerical fluxes. The latter must expose the following types:
 * - `target_fields`: A `tmpl::list` of databox tags for which the fluxes are
 * computed. This is needed to apply the computed and lifted fluxes to the
 * fields in the databox. There must be a `Variables` with precisely these tags
 * in the databox.
 * - `argument_tags`: A `tmpl::list` of tags of which interface tags
 * must exist in the databox. When sending fluxes, the values of these interface
 * tags are passed to the `package_data` function to provide the necessary data.
 * - `package_tags`: A `tmpl::list` of databox tags that the `package_data`
 * function computes. This data is communicated and made available to the
 * numerical flux computer on both sides of the boundary. It is passed to the
 * `()` operator when fluxes are applied, e.g. in
 * `dg::ApplyBoundaryFluxesGlobalTimestepping`.
 */
template <size_t Dim, typename TemporalIdTag, typename NumericalFluxTag>
struct FluxCommunicationTypes {
 public:
  /// The type that computes numerical fluxes
  using numerical_flux = typename NumericalFluxTag::type;

  /// The type of the Variables sent to neighboring elements.
  using PackagedData = Variables<typename numerical_flux::package_tags>;

  /// The Variables tag that stores the normal numerical fluxes. Computed by the
  /// numerical flux computer and used e.g. by lift_flux.
  using normal_dot_numerical_fluxes_tag = db::add_tag_prefix<
      Tags::NormalDotNumericalFlux,
      Tags::Variables<typename numerical_flux::target_fields>>;

  // The Variables tag that stores the normal fluxes, also used e.g. by
  // lift_flux to compute the strong DG flux contribution on an interface. These
  // have to be computed on the interfaces and stored in the databox before
  // SendDataForFluxes is called.
  using normal_dot_fluxes_tag = db::add_tag_prefix<
      Tags::NormalDotFlux,
      Tags::Variables<typename numerical_flux::target_fields>>;

  /// Variables tag for storing the magnitude of the face normal in
  /// the LocalData.
  struct MagnitudeOfFaceNormal {
    using type = Scalar<DataVector>;
  };

  /// The type of the Variables needed for the local part of the
  /// numerical flux computations.  Contains the PackagedData, the
  /// normal fluxes, and MagnitudeOfFaceNormal.
  using LocalData = Variables<tmpl::remove_duplicates<tmpl::append<
      typename db::item_type<normal_dot_fluxes_tag>::tags_list,
      typename PackagedData::tags_list, tmpl::list<MagnitudeOfFaceNormal>>>>;

 private:
  template <typename Base, typename Tag, size_t VolumeDim>
  struct BasedMortars : Base, Tags::Mortars<Tag, VolumeDim> {};

 public:
  /// The DataBox tag for the data stored on the mortars for global
  /// time stepping.
  using global_time_stepping_mortar_data_tag =
      BasedMortars<Tags::VariablesBoundaryData,
                   Tags::SimpleBoundaryData<db::item_type<TemporalIdTag>,
                                            LocalData, PackagedData>,
                   Dim>;

  /// The inbox tag for flux communication.
  struct FluxesTag {
    using temporal_id = typename db::item_type<TemporalIdTag>;
    using type =
        std::map<temporal_id,
                 std::unordered_map<
                     std::pair<Direction<Dim>, ElementId<Dim>>,
                     std::pair<temporal_id, PackagedData>,
                     boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>>;
  };
};
}  // namespace dg
