// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FixConservatives.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Evolution/VariableFixing/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace EquationsOfState {
template <bool IsRelativistic, size_t ThermodynamicDim>
class EquationOfState;
}  // namespace EquationsOfState
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
template <typename TagsList>
class Variables;
/// \endcond

namespace grmhd::ValenciaDivClean::subcell {
/*!
 * \brief Fix the conservative variables and compute the primitive variables.
 *
 * Sets `ValenciaDivClean::Tags::VariablesNeededFixing` to `true` if the
 * conservative variables needed fixing, otherwise sets the tag to `false`.
 */
template <typename OrderedListOfRecoverySchemes>
struct FixConservativesAndComputePrims {
  using return_tags = tmpl::list<ValenciaDivClean::Tags::VariablesNeededFixing,
                                 typename System::variables_tag,
                                 typename System::primitive_variables_tag>;
  using argument_tags = tmpl::list<
      ::Tags::VariableFixer<grmhd::ValenciaDivClean::FixConservatives>,
      hydro::Tags::EquationOfStateBase,
      gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>,
      gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataVector>,
      gr::Tags::SqrtDetSpatialMetric<DataVector>>;

  template <size_t ThermodynamicDim>
  static void apply(
      gsl::not_null<bool*> needed_fixing,
      gsl::not_null<typename System::variables_tag::type*> conserved_vars_ptr,
      gsl::not_null<Variables<hydro::grmhd_tags<DataVector>>*>
          primitive_vars_ptr,
      const grmhd::ValenciaDivClean::FixConservatives& fix_conservatives,
      const EquationsOfState::EquationOfState<true, ThermodynamicDim>& eos,
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
      const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
      const Scalar<DataVector>& sqrt_det_spatial_metric);
};
}  // namespace grmhd::ValenciaDivClean::subcell
