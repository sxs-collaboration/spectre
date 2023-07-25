// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <type_traits>

#include "DataStructures/DataBox/Protocols/Mutator.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/CreateIsCallable.hpp"

namespace ForceFree {
namespace detail {
CREATE_IS_CALLABLE(interior_mask)
CREATE_IS_CALLABLE_V(interior_mask)
CREATE_IS_CALLABLE_R_V(interior_mask)
}  // namespace detail

/*!
 * \brief Assign the masking scalar variable (see Tags::NsInteriorMask) at the
 * initialization phase in NS magnetosphere simulations.
 *
 * Run the `interior_mask()` member function of the initial data if it is
 * callable.
 */
template <typename Metavariables, bool UsedForFdGrid>
struct MaskNeutronStarInterior : tt::ConformsTo<db::protocols::Mutator> {
  using argument_tags = tmpl::list<
      tmpl::conditional_t<
          UsedForFdGrid,
          evolution::dg::subcell::Tags::Coordinates<3, Frame::Inertial>,
          domain::Tags::Coordinates<3, Frame::Inertial>>,
      evolution::initial_data::Tags::InitialData>;

  using return_tags = tmpl::list<tmpl::conditional_t<
      UsedForFdGrid,
      evolution::dg::subcell::Tags::Inactive<Tags::NsInteriorMask>,
      Tags::NsInteriorMask>>;

  static void apply(
      const gsl::not_null<std::optional<Scalar<DataVector>>*>
          neutron_star_interior_mask,
      const tnsr::I<DataVector, 3, Frame::Inertial>& inertial_coords,
      const evolution::initial_data::InitialData& solution_or_data) {
    using all_data_and_solutions =
        tmpl::at<typename Metavariables::factory_creation::factory_classes,
                 evolution::initial_data::InitialData>;

    call_with_dynamic_type<void, all_data_and_solutions>(
        &solution_or_data, [&neutron_star_interior_mask,
                            &inertial_coords](const auto* initial_data_ptr) {
          using InitialData = std::decay_t<decltype(*initial_data_ptr)>;

          if constexpr (detail::is_interior_mask_callable_r_v<
                            std::optional<Scalar<DataVector>>, InitialData,
                            tnsr::I<DataVector, 3, Frame::Inertial>>) {
            (*neutron_star_interior_mask) =
                (*initial_data_ptr).interior_mask(inertial_coords);
          }
        });
  }
};

}  // namespace ForceFree
