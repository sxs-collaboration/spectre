// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <pup.h>
#include <pup_stl.h>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"

/// \cond
class DataVector;
namespace gsl {
template <class T>
class not_null;
}  // namespace gsl
namespace EquationsOfState {
template <bool IsRelativistic, size_t ThermodynamicDim>
class EquationOfState;
}  // namespace EquationsOfState
/// \endcond

namespace NewtonianEuler {
/*!
 * Holds classes implementing sources for the Newtonian Euler system.
 */
namespace Sources {
/*!
 * \brief Source terms base class.
 */
template <size_t Dim>
class Source : public PUP::able {
 protected:
  Source() = default;

 public:
  ~Source() override = default;

  /// \cond
  explicit Source(CkMigrateMessage* msg) : PUP::able(msg) {}
  WRAPPED_PUPable_abstract(Source);
  /// \endcond

  virtual auto get_clone() const -> std::unique_ptr<Source> = 0;

  virtual void operator()(
      gsl::not_null<Scalar<DataVector>*> source_mass_density_cons,
      gsl::not_null<tnsr::I<DataVector, Dim>*> source_momentum_density,
      gsl::not_null<Scalar<DataVector>*> source_energy_density,
      const Scalar<DataVector>& mass_density_cons,
      const tnsr::I<DataVector, Dim>& momentum_density,
      const Scalar<DataVector>& energy_density,
      const tnsr::I<DataVector, Dim>& velocity,
      const Scalar<DataVector>& pressure,
      const Scalar<DataVector>& specific_internal_energy,
      const EquationsOfState::EquationOfState<false, 2>& eos,
      const tnsr::I<DataVector, Dim>& coords, double time) const = 0;
};
}  // namespace Sources
}  // namespace NewtonianEuler
