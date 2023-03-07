// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Tags.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Background.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Xcts::Tags {

/*!
 * \brief MHD quantities retrieved from the background solution/data
 *
 * Retrieve the `HydroTags` from the background solution/data so they can be
 * written out to disk. These quantities don't take part in a pure XCTS solve,
 * (which only solves for the gravity sector given the matter profile) except
 * for their contributions to the matter source terms in the XCTS equations.
 * However, the matter source terms are computed and stored separately, so these
 * hydro quantities are only used for observations.
 */
template <typename HydroTags>
struct HydroQuantitiesCompute : ::Tags::Variables<HydroTags>, db::ComputeTag {
  using base = ::Tags::Variables<HydroTags>;
  using argument_tags = tmpl::list<
      domain::Tags::Coordinates<3, Frame::Inertial>,
      elliptic::Tags::Background<elliptic::analytic_data::Background>,
      Parallel::Tags::Metavariables>;
  template <typename Metavariables>
  static void function(const gsl::not_null<Variables<HydroTags>*> result,
                       const tnsr::I<DataVector, 3>& inertial_coords,
                       const elliptic::analytic_data::Background& background,
                       const Metavariables& /*meta*/) {
    using background_classes =
        tmpl::at<typename Metavariables::factory_creation::factory_classes,
                 elliptic::analytic_data::Background>;
    *result = call_with_dynamic_type<Variables<HydroTags>, background_classes>(
        &background, [&inertial_coords](const auto* const derived) {
          return variables_from_tagged_tuple(
              derived->variables(inertial_coords, HydroTags{}));
        });
  }
};

}  // namespace Xcts::Tags
