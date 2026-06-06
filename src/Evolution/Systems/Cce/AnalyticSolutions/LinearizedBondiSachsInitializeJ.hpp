// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <complex>
#include <cstddef>
#include <limits>
#include <memory>

#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Cce/Initialize/InitializeJ.hpp"
#include "Parallel/NodeLock.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"

/// \cond
class DataVector;
class ComplexDataVector;
/// \endcond

namespace Cce::Solutions::LinearizedBondiSachs_detail::InitializeJ {
// First hypersurface Initialization for the
// `Cce::Solutions::LinearizedBondiSachs` analytic solution.
//
// This initialization procedure should not be used except when the
// `Cce::Solutions::LinearizedBondiSachs` analytic solution is used,
// as a consequence, this initial data generator is deliberately not
// option-creatable; it should only be obtained from the `get_initialize_j`
// function of `Cce::InitializeJ::LinearizedBondiSachs`.
//
// It lives in its own lightweight header (depending only on `InitializeJ.hpp`,
// not the heavy `LinearizedBondiSachs.hpp`/`SphericalMetricData.hpp`) so that
// `InitializeJ.hpp` can bottom-include it for the `call_with_dynamic_type`
// dispatch over `InitializeJ<false>::creatable_classes` without an include
// cycle.
struct LinearizedBondiSachs : ::Cce::InitializeJ::InitializeJ<false> {
  WRAPPED_PUPable_decl_template(LinearizedBondiSachs);  // NOLINT
  explicit LinearizedBondiSachs(CkMigrateMessage* /*unused*/) {}

  LinearizedBondiSachs() = default;

  LinearizedBondiSachs(double start_time, double frequency,
                       std::complex<double> c_2a, std::complex<double> c_2b,
                       std::complex<double> c_3a, std::complex<double> c_3b);

  // Deliberately not option-creatable (see note above); obtained only via
  // `Cce::Solutions::LinearizedBondiSachs::get_initialize_j`. It is still in
  // `InitializeJ<false>::creatable_classes` so the dispatch and charm
  // registration see it; `factory_creatable = false` keeps it out of the
  // option factory.
  static constexpr bool factory_creatable = false;

  std::unique_ptr<InitializeJ> get_clone() const override;

  void operator()(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
      gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
      gsl::not_null<
          tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
          angular_cauchy_coordinates,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_j,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_dr_j,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& r,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& beta, size_t l_max,
      size_t number_of_radial_points,
      gsl::not_null<Parallel::NodeLock*> hdf5_lock) const;

  void pup(PUP::er& /*p*/) override;

 private:
  std::complex<double> c_2a_ = std::numeric_limits<double>::signaling_NaN();
  std::complex<double> c_2b_ = std::numeric_limits<double>::signaling_NaN();
  std::complex<double> c_3a_ = std::numeric_limits<double>::signaling_NaN();
  std::complex<double> c_3b_ = std::numeric_limits<double>::signaling_NaN();
  double frequency_ = std::numeric_limits<double>::signaling_NaN();
  double time_ = std::numeric_limits<double>::signaling_NaN();
};
}  // namespace Cce::Solutions::LinearizedBondiSachs_detail::InitializeJ
