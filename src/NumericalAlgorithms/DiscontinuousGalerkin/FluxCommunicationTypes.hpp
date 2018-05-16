// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/functional/hash.hpp>  // IWYU pragma: keep
#include <cstddef>
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
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace dg {
/// \ingroup DiscontinuousGalerkinGroup
/// \brief Types related to flux communication.
template <typename Metavariables>
struct FluxCommunicationTypes {
 private:
  using system = typename Metavariables::system;
  static constexpr size_t volume_dim = system::volume_dim;

 public:
  /// The type of the Variables sent to neighboring elements.
  using PackagedData = Variables<
      typename Metavariables::normal_dot_numerical_flux::type::package_tags>;

  /// The DataBox tag for the normal fluxes of the evolved variables.
  using normal_dot_fluxes_tag =
      db::add_tag_prefix<Tags::NormalDotFlux, typename system::variables_tag>;

  /// Variables tag for storing the magnitude of the face normal in
  /// the LocalData.
  struct MagnitudeOfFaceNormal {
    using type = Scalar<DataVector>;
  };

  /// The type of the Variables needed for the local part of the flux
  /// numerical flux computations.  Contains the PackagedData, the
  /// normal fluxes, and MagnitudeOfFaceNormal.
  using LocalData = Variables<tmpl::remove_duplicates<tmpl::append<
      typename db::item_type<normal_dot_fluxes_tag>::tags_list,
      typename PackagedData::tags_list, tmpl::list<MagnitudeOfFaceNormal>>>>;

  /// The DataBox tag for the data stored on the mortars.
  using mortar_data_tag =
      Tags::Mortars<Tags::SimpleBoundaryData<
                        db::item_type<typename Metavariables::temporal_id>,
                        LocalData, PackagedData>,
                    volume_dim>;

  /// The inbox tag for flux communication.
  struct FluxesTag {
    using temporal_id = db::item_type<typename Metavariables::temporal_id>;
    using type = std::unordered_map<
        temporal_id,
        std::unordered_map<
            std::pair<Direction<volume_dim>, ElementId<volume_dim>>,
            PackagedData,
            boost::hash<
                std::pair<Direction<volume_dim>, ElementId<volume_dim>>>>>;
  };
};
}  // namespace dg
