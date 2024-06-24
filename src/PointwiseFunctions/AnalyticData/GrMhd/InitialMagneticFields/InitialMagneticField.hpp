// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <pup.h>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;

namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

namespace grmhd::AnalyticData {

/*!
 * \brief Things related to the initial (seed) magnetic field that can be
 * superposed on GRMHD initial data.
 *
 * In many cases we assign magnetic fields in terms of the vector potential,
 * which can be computed as
 *
 * \f{align*}{
 *   B^i & = n_a\epsilon^{aijk}\partial_jA_k \\
 *       & = \frac{1}{\sqrt{\gamma}}[ijk]\partial_j A_k,
 * \f}
 *
 * where \f$[ijk]\f$ is the total anti-symmetric symbol.
 *
 * For example, in the Cartesian coordinates,
 *
 * \f{align*}{
 *   B^x & = \frac{1}{\sqrt{\gamma}} (\partial_y A_z - \partial_z A_y), \\
 *   B^y & = \frac{1}{\sqrt{\gamma}} (\partial_z A_x - \partial_x A_z), \\
 *   B^z & = \frac{1}{\sqrt{\gamma}} (\partial_x A_y - \partial_y A_x).
 * \f}
 *
 *
 * \warning The magnetic field classes assume the magnetic field is initialized,
 * both in size and value, before being passed into the `variables` function.
 * This is so that multiple magnetic fields can be superposed. Each magnetic
 * field configuration does a `+=` to make this possible.
 */
namespace InitialMagneticFields {

/*!
 * \brief The abstract base class for initial magnetic field configurations.
 *
 * \warning This assumes the magnetic field is initialized, both in size and
 * value, before being passed into the `variables` function. This is so that
 * multiple magnetic fields can be superposed. Each magnetic field
 * configuration does a `+=` to make this possible.
 */
class InitialMagneticField : public PUP::able {
 protected:
  InitialMagneticField() = default;

 public:
  ~InitialMagneticField() override = default;

  virtual auto get_clone() const -> std::unique_ptr<InitialMagneticField> = 0;

  virtual void variables(
      gsl::not_null<tnsr::I<DataVector, 3>*> result,
      const tnsr::I<DataVector, 3>& coords, const Scalar<DataVector>& pressure,
      const Scalar<DataVector>& sqrt_det_spatial_metric,
      const tnsr::i<DataVector, 3>& deriv_pressure) const = 0;

  virtual void variables(gsl::not_null<tnsr::I<double, 3>*> result,
                         const tnsr::I<double, 3>& coords,
                         const Scalar<double>& pressure,
                         const Scalar<double>& sqrt_det_spatial_metric,
                         const tnsr::i<double, 3>& deriv_pressure) const = 0;

  virtual bool is_equal(const InitialMagneticField& rhs) const = 0;

  /// \cond
  explicit InitialMagneticField(CkMigrateMessage* msg) : PUP::able(msg) {}
  WRAPPED_PUPable_abstract(InitialMagneticField);
  /// \endcond
};

}  // namespace InitialMagneticFields
}  // namespace grmhd::AnalyticData
