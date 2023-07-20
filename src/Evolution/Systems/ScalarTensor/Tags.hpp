// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Options/String.hpp"
#include "Utilities/TMPL.hpp"

/*!
 * \brief Tags for the scalar tensor system.
 */
namespace ScalarTensor {
namespace Tags {
/*!
 * \brief Represents the trace-reversed stress-energy tensor of the scalar
 * field.
 */
template <typename DataType, size_t Dim, typename Fr = Frame::Inertial>
struct TraceReversedStressEnergy : db::SimpleTag {
  using type = tnsr::aa<DataType, Dim, Fr>;
};

/*!
 * \brief Tag holding the source term of the scalar equation.
 *
 * \details This tag hold the source term \f$ \mathcal{S} \f$,
 * entering a wave equation of the form
 * \f[
 *   \Box \Psi = \mathcal{S} ~.
 * \f]
 */
struct ScalarSource : db::SimpleTag {
  using type = Scalar<DataVector>;
};

}  // namespace Tags

namespace OptionTags {
/*!
 * \brief Scalar mass parameter.
 */
struct ScalarMass {
  static std::string name() { return "ScalarMass"; }
  using type = double;
  static constexpr Options::String help{
      "Mass of the scalar field in code units"};
};
}  // namespace OptionTags

namespace Tags {
struct ScalarMass : db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<OptionTags::ScalarMass>;
  static constexpr bool pass_metavariables = false;
  static double create_from_options(const double mass_psi) { return mass_psi; }
};
}  // namespace Tags

}  // namespace ScalarTensor
