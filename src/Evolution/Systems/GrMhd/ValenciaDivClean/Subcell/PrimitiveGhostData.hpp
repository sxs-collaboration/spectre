// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/VariablesTag.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
template <typename T>
class Variables;
/// \endcond

namespace grmhd::ValenciaDivClean::subcell {
/*!
 * \brief Computes the rest mass density \f$\rho\f$, electron fraction
 * \f$Y_e\f$, pressure \f$p\f$, Lorentz factor times the spatial velocity \f$W
 * v^i\f$, magnetic field \f$B^i\f$, and the divergence cleaning field
 * \f$\Phi\f$ so they can be used for reconstruction.
 *
 * The computation copies the data from the primitive variables to a new
 * Variables and computes \f$W v^i\f$. In the future we will likely want to
 * elide this copy but that requires support from the actions.
 *
 * This mutator is passed to
 * `evolution::dg::subcell::Actions::SendDataForReconstruction`.
 */
class PrimitiveGhostVariables {
 private:
  using prims_to_reconstruct_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::ElectronFraction<DataVector>,
                 hydro::Tags::Pressure<DataVector>,
                 hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>,
                 hydro::Tags::MagneticField<DataVector, 3>,
                 hydro::Tags::DivergenceCleaningField<DataVector>>;

 public:
  using return_tags = tmpl::list<>;
  using argument_tags =
      tmpl::list<::Tags::Variables<hydro::grmhd_tags<DataVector>>>;

  static DataVector apply(const Variables<hydro::grmhd_tags<DataVector>>& prims,
                          size_t rdmp_size);
};
}  // namespace grmhd::ValenciaDivClean::subcell
