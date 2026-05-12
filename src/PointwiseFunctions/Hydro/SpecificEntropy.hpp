// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace EquationsOfState {
template <bool IsRelativistic, size_t ThermodynamicDim>
class EquationOfState;
}  // namespace EquationsOfState
/// \endcond

namespace hydro {
/// @{
/*!
 * \ingroup EquationsOfStateGroup
 * \brief Computes the specific entropy
 */
template <typename DataType, size_t ThermodynamicDim>
void specific_entropy(
    gsl::not_null<Scalar<DataType>*> result,
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& temperature,
    const Scalar<DataType>& electron_fraction,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state);

template <typename DataType, size_t ThermodynamicDim>
Scalar<DataType> specific_entropy(
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& temperature,
    const Scalar<DataType>& electron_fraction,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state);
/// @}

namespace Tags {
/// Compute item for the specific entropy \f$s\f$.
/// \see hydro::specific_entropy
///
/// Can be retrieved using `hydro::Tags::SpecificEntropy`
template <typename DataType>
struct SpecificEntropyCompute : SpecificEntropy<DataType>, db::ComputeTag {
  using argument_tags =
      typename tmpl::list<RestMassDensity<DataType>, Temperature<DataType>,
                          ElectronFraction<DataType>,
                          hydro::Tags::GrmhdEquationOfState>;

  using return_type = Scalar<DataType>;

  template <typename EquationOfStateType>
  static void function(const gsl::not_null<Scalar<DataType>*> result,
                       const Scalar<DataType>& rest_mass_density,
                       const Scalar<DataType>& temperature,
                       const Scalar<DataType>& electron_fraction,
                       const EquationOfStateType& equation_of_state) {
    specific_entropy(result, rest_mass_density, temperature, electron_fraction,
                     equation_of_state);
  }

  using base = SpecificEntropy<DataType>;
};
}  // namespace Tags
}  // namespace hydro
