// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"  // IWYU pragma: keep
#include "Utilities/TMPL.hpp"

/// \cond
namespace gsl {
template <class T>
class not_null;
}  // namespace gsl
class DataVector;
/// \endcond

namespace RadiationTransport {
namespace M1Grey {

/*!
 * \brief Compute the characteristic speeds for the M1 system
 *
 * At this point, for testing purposes, we just set all
 * speeds to 1... this needs to be fixed in the future to
 * use the correct speed values.
 */
void characteristic_speeds(
    gsl::not_null<std::array<DataVector, 4>*> pchar_speeds,
    const Scalar<DataVector>& lapse);

namespace Tags {
/// \brief Compute the characteristic speeds for the M1 system
///
struct CharacteristicSpeedsCompute : Tags::CharacteristicSpeeds,
                                     db::ComputeTag {
  using base = Tags::CharacteristicSpeeds;
  using argument_tags = tmpl::list<gr::Tags::Lapse<>>;

  using return_type = std::array<DataVector, 4>;

  static constexpr auto function = characteristic_speeds;
};
}  // namespace Tags

struct ComputeLargestCharacteristicSpeed {
  using argument_tags = tmpl::list<>;
  static double apply() { return 1.0; }
};

}  // namespace M1Grey
}  // namespace RadiationTransport
