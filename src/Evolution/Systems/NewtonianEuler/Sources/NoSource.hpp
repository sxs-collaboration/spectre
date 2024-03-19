// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/NewtonianEuler/Sources/Source.hpp"
#include "Options/String.hpp"
#include "Utilities/TMPL.hpp"

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
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace NewtonianEuler::Sources {
/*!
 * \brief Used to mark that the initial data do not require source
 * terms in the evolution equations.
 */
template <size_t Dim>
class NoSource : public Source<Dim> {
 public:
  using options = tmpl::list<>;

  static constexpr Options::String help = {"No source terms added."};

  NoSource() = default;
  NoSource(const NoSource& /*rhs*/) = default;
  NoSource& operator=(const NoSource& /*rhs*/) = default;
  NoSource(NoSource&& /*rhs*/) = default;
  NoSource& operator=(NoSource&& /*rhs*/) = default;
  ~NoSource() override = default;

  /// \cond
  explicit NoSource(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(NoSource);
  /// \endcond

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

  auto get_clone() const -> std::unique_ptr<Source<Dim>> override;

  void operator()(
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
      const tnsr::I<DataVector, Dim>& coords, double time) const override;

  using sourced_variables = tmpl::list<>;
  using argument_tags = tmpl::list<>;
};
}  // namespace NewtonianEuler::Sources
