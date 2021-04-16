// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/Xcts/Equations.hpp"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <optional>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Xcts/Geometry.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace Xcts {
namespace detail {
void fully_contract_flat_cartesian(
    const gsl::not_null<Scalar<DataVector>*> result,
    const tnsr::II<DataVector, 3>& tensor1,
    const tnsr::II<DataVector, 3>& tensor2) noexcept {
  get(*result) = tensor1.multiplicity(0_st) * tensor1[0] * tensor2[0];
  for (size_t i = 1; i < tensor1.size(); ++i) {
    get(*result) += tensor1.multiplicity(i) * tensor1[i] * tensor2[i];
  }
}
void fully_contract(const gsl::not_null<Scalar<DataVector>*> result,
                    const gsl::not_null<DataVector*> buffer1,
                    const gsl::not_null<DataVector*> buffer2,
                    const tnsr::II<DataVector, 3>& tensor1,
                    const tnsr::II<DataVector, 3>& tensor2,
                    const tnsr::ii<DataVector, 3>& metric) noexcept {
  get(*result) = 0.;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      *buffer1 = metric.get(i, 0) * tensor1.get(0, j);
      *buffer2 = metric.get(j, 0) * tensor2.get(i, 0);
      for (size_t k = 1; k < 3; ++k) {
        *buffer1 += metric.get(i, k) * tensor1.get(k, j);
        *buffer2 += metric.get(j, k) * tensor2.get(i, k);
      }
      get(*result) += *buffer1 * *buffer2;
    }
  }
}
}  // namespace detail

template <int ConformalMatterScale>
void add_hamiltonian_sources(
    const gsl::not_null<Scalar<DataVector>*> hamiltonian_constraint,
    const Scalar<DataVector>& conformal_energy_density,
    const Scalar<DataVector>& extrinsic_curvature_trace,
    const Scalar<DataVector>& conformal_factor) noexcept {
  // See BaumgarteShapiro, Eq. (3.110)
  get(*hamiltonian_constraint) +=
      (square(get(extrinsic_curvature_trace)) / 12. -
       2. * M_PI * get(conformal_energy_density) /
           pow<ConformalMatterScale>(get(conformal_factor))) *
      pow<5>(get(conformal_factor));
}

template <int ConformalMatterScale>
void add_linearized_hamiltonian_sources(
    const gsl::not_null<Scalar<DataVector>*> linearized_hamiltonian_constraint,
    const Scalar<DataVector>& conformal_energy_density,
    const Scalar<DataVector>& extrinsic_curvature_trace,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& conformal_factor_correction) noexcept {
  get(*linearized_hamiltonian_constraint) +=
      ((5. / 12.) * square(get(extrinsic_curvature_trace)) -
       2. * (5. - ConformalMatterScale) * M_PI * get(conformal_energy_density) /
           pow<ConformalMatterScale>(get(conformal_factor))) *
      pow<4>(get(conformal_factor)) * get(conformal_factor_correction);
}

void add_distortion_hamiltonian_sources(
    const gsl::not_null<Scalar<DataVector>*> hamiltonian_constraint,
    const Scalar<DataVector>&
        longitudinal_shift_minus_dt_conformal_metric_over_lapse_square,
    const Scalar<DataVector>& conformal_factor) noexcept {
  // See BaumgarteShapiro, Eq. (3.110) and (3.112). Note: 0.03125 = 1 / 32
  get(*hamiltonian_constraint) -=
      0.03125 * pow<5>(get(conformal_factor)) *
      get(longitudinal_shift_minus_dt_conformal_metric_over_lapse_square);
}

void add_linearized_distortion_hamiltonian_sources(
    const gsl::not_null<Scalar<DataVector>*> linearized_hamiltonian_constraint,
    const Scalar<DataVector>&
        longitudinal_shift_minus_dt_conformal_metric_over_lapse_square,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& conformal_factor_correction) noexcept {
  // Note: 0.15625 = 5 / 32
  get(*linearized_hamiltonian_constraint) -=
      0.15625 * pow<4>(get(conformal_factor)) *
      get(longitudinal_shift_minus_dt_conformal_metric_over_lapse_square) *
      get(conformal_factor_correction);
}

void add_curved_hamiltonian_or_lapse_sources(
    const gsl::not_null<Scalar<DataVector>*> hamiltonian_or_lapse_equation,
    const Scalar<DataVector>& conformal_ricci_scalar,
    const Scalar<DataVector>& field) noexcept {
  // See BaumgarteShapiro, Eq. (3.110) and (3.111). Note: 0.125 = 1 / 8
  get(*hamiltonian_or_lapse_equation) +=
      0.125 * get(conformal_ricci_scalar) * get(field);
}

template <int ConformalMatterScale>
void add_lapse_sources(
    const gsl::not_null<Scalar<DataVector>*> lapse_equation,
    const Scalar<DataVector>& conformal_energy_density,
    const Scalar<DataVector>& conformal_stress_trace,
    const Scalar<DataVector>& extrinsic_curvature_trace,
    const Scalar<DataVector>& dt_extrinsic_curvature_trace,
    const Scalar<DataVector>& shift_dot_deriv_extrinsic_curvature_trace,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor) noexcept {
  // See BaumgarteShapiro, Eq. (3.111)
  get(*lapse_equation) +=
      get(lapse_times_conformal_factor) * pow<4>(get(conformal_factor)) *
          ((5. / 12.) * square(get(extrinsic_curvature_trace)) +
           2. * M_PI *
               (get(conformal_energy_density) +
                2. * get(conformal_stress_trace)) /
               pow<ConformalMatterScale>(get(conformal_factor))) +
      pow<5>(get(conformal_factor)) *
          (get(shift_dot_deriv_extrinsic_curvature_trace) -
           get(dt_extrinsic_curvature_trace));
}

template <int ConformalMatterScale>
void add_linearized_lapse_sources(
    const gsl::not_null<Scalar<DataVector>*> linearized_lapse_equation,
    const Scalar<DataVector>& conformal_energy_density,
    const Scalar<DataVector>& conformal_stress_trace,
    const Scalar<DataVector>& extrinsic_curvature_trace,
    const Scalar<DataVector>& dt_extrinsic_curvature_trace,
    const Scalar<DataVector>& shift_dot_deriv_extrinsic_curvature_trace,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const Scalar<DataVector>& conformal_factor_correction,
    const Scalar<DataVector>&
        lapse_times_conformal_factor_correction) noexcept {
  get(*linearized_lapse_equation) +=
      pow<3>(get(conformal_factor)) *
          ((get(conformal_factor) *
                get(lapse_times_conformal_factor_correction) +
            4. * get(lapse_times_conformal_factor) *
                get(conformal_factor_correction)) *
               5. / 12. * square(get(extrinsic_curvature_trace)) +
           (get(conformal_factor) *
                get(lapse_times_conformal_factor_correction) +
            (4. - ConformalMatterScale) * get(lapse_times_conformal_factor) *
                get(conformal_factor_correction)) *
               2. * M_PI *
               (get(conformal_energy_density) +
                2 * get(conformal_stress_trace)) /
               pow<ConformalMatterScale>(get(conformal_factor))) +
      5. * pow<4>(get(conformal_factor)) * get(conformal_factor_correction) *
          (get(shift_dot_deriv_extrinsic_curvature_trace) -
           get(dt_extrinsic_curvature_trace));
}

void add_distortion_hamiltonian_and_lapse_sources(
    gsl::not_null<Scalar<DataVector>*> hamiltonian_constraint,
    gsl::not_null<Scalar<DataVector>*> lapse_equation,
    const Scalar<DataVector>&
        longitudinal_shift_minus_dt_conformal_metric_square,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor) noexcept {
  // See BaumgarteShapiro, Eq. (3.110--3.112). Note: 0.03125 = 1 / 32 and
  // 0.21875 = 7 / 32
  get(*hamiltonian_constraint) -=
      0.03125 * pow<7>(get(conformal_factor)) /
      square(get(lapse_times_conformal_factor)) *
      get(longitudinal_shift_minus_dt_conformal_metric_square);
  get(*lapse_equation) +=
      0.21875 * pow<6>(get(conformal_factor)) /
      get(lapse_times_conformal_factor) *
      get(longitudinal_shift_minus_dt_conformal_metric_square);
}

void add_linearized_distortion_hamiltonian_and_lapse_sources(
    gsl::not_null<Scalar<DataVector>*> hamiltonian_constraint,
    gsl::not_null<Scalar<DataVector>*> lapse_equation,
    const Scalar<DataVector>&
        longitudinal_shift_minus_dt_conformal_metric_square,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const Scalar<DataVector>& conformal_factor_correction,
    const Scalar<DataVector>&
        lapse_times_conformal_factor_correction) noexcept {
  // Note: 0.03125 = 1 / 32 and 0.21875 = 7 / 32
  get(*hamiltonian_constraint) -=
      0.03125 * pow<6>(get(conformal_factor)) *
      (7. * get(conformal_factor_correction) -
       2. * get(conformal_factor) / get(lapse_times_conformal_factor) *
           get(lapse_times_conformal_factor_correction)) /
      square(get(lapse_times_conformal_factor)) *
      get(longitudinal_shift_minus_dt_conformal_metric_square);
  get(*lapse_equation) +=
      0.21875 * pow<5>(get(conformal_factor)) *
      (6. * get(conformal_factor_correction) -
       get(conformal_factor) / get(lapse_times_conformal_factor) *
           get(lapse_times_conformal_factor_correction)) /
      get(lapse_times_conformal_factor) *
      get(longitudinal_shift_minus_dt_conformal_metric_square);
}

template <int ConformalMatterScale, Geometry ConformalGeometry>
void add_momentum_sources_impl(
    const gsl::not_null<Scalar<DataVector>*> hamiltonian_constraint,
    const gsl::not_null<Scalar<DataVector>*> lapse_equation,
    const gsl::not_null<tnsr::I<DataVector, 3>*> momentum_constraint,
    const tnsr::I<DataVector, 3>& conformal_momentum_density,
    const tnsr::i<DataVector, 3>& extrinsic_curvature_trace_gradient,
    const std::optional<std::reference_wrapper<const tnsr::ii<DataVector, 3>>>
        conformal_metric,
    const std::optional<std::reference_wrapper<const tnsr::II<DataVector, 3>>>
        inv_conformal_metric,
    const tnsr::I<DataVector, 3>& minus_div_dt_conformal_metric,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const tnsr::I<DataVector, 3>& conformal_factor_flux,
    const tnsr::I<DataVector, 3>& lapse_times_conformal_factor_flux,
    const tnsr::II<DataVector, 3>&
        longitudinal_shift_minus_dt_conformal_metric) noexcept {
  if constexpr (ConformalGeometry == Geometry::FlatCartesian) {
    ASSERT(not conformal_metric.has_value() and
               not inv_conformal_metric.has_value(),
           "You don't need to pass a conformal metric to this function when it "
           "is specialized for a flat conformal geometry in Cartesian "
           "coordinates.");
  } else {
    ASSERT(conformal_metric.has_value() and inv_conformal_metric.has_value(),
           "You must pass a conformal metric to this function when it is "
           "specialized for a curved conformal geometry.");
  }
  const size_t num_points = get(conformal_factor).size();
  TempBuffer<tmpl::list<::Tags::TempScalar<0>, ::Tags::TempI<1, 3>,
                        ::Tags::Tempi<2, 3>>>
      buffer{num_points};
  {
    auto& longitudinal_shift_square = get<::Tags::TempScalar<0>>(buffer);
    if constexpr (ConformalGeometry == Geometry::FlatCartesian) {
      detail::fully_contract_flat_cartesian(
          make_not_null(&longitudinal_shift_square),
          longitudinal_shift_minus_dt_conformal_metric,
          longitudinal_shift_minus_dt_conformal_metric);
    } else {
      auto& buffer1 = get<0>(get<::Tags::TempI<1, 3>>(buffer));
      auto& buffer2 = get<1>(get<::Tags::TempI<1, 3>>(buffer));
      detail::fully_contract(make_not_null(&longitudinal_shift_square),
                             make_not_null(&buffer1), make_not_null(&buffer2),
                             longitudinal_shift_minus_dt_conformal_metric,
                             longitudinal_shift_minus_dt_conformal_metric,
                             conformal_metric->get());
    }
    // Add shift terms to Hamiltonian and lapse equations, see BaumgarteShapiro,
    // Eq. (3.110--3.112). Note: 0.03125 = 1 / 32 and 0.21875 = 7 / 32
    get(*hamiltonian_constraint) -= 0.03125 * pow<7>(get(conformal_factor)) /
                                    square(get(lapse_times_conformal_factor)) *
                                    get(longitudinal_shift_square);
    get(*lapse_equation) += 0.21875 * pow<6>(get(conformal_factor)) /
                            get(lapse_times_conformal_factor) *
                            get(longitudinal_shift_square);
  }
  // Add sources to momentum constraint, see BaumgarteShapiro, Eq. (3.109)
  {
    // Extrinsic curvature term
    auto& extrinsic_curvature_trace_gradient_term =
        get<::Tags::TempI<1, 3>>(buffer);
    if constexpr (ConformalGeometry == Geometry::FlatCartesian) {
      get<0>(extrinsic_curvature_trace_gradient_term) =
          get<0>(extrinsic_curvature_trace_gradient);
      get<1>(extrinsic_curvature_trace_gradient_term) =
          get<1>(extrinsic_curvature_trace_gradient);
      get<2>(extrinsic_curvature_trace_gradient_term) =
          get<2>(extrinsic_curvature_trace_gradient);
    } else {
      raise_or_lower_index(
          make_not_null(&extrinsic_curvature_trace_gradient_term),
          extrinsic_curvature_trace_gradient, inv_conformal_metric->get());
    }
    for (size_t i = 0; i < 3; ++i) {
      extrinsic_curvature_trace_gradient_term.get(i) *=
          4. / 3. * get(lapse_times_conformal_factor) / get(conformal_factor);
      momentum_constraint->get(i) +=
          extrinsic_curvature_trace_gradient_term.get(i);
    }
  }
  {
    // Lapse deriv term, to be contracted with longitudinal shift
    auto& lapse_deriv_term = get<::Tags::TempI<1, 3>>(buffer);
    for (size_t i = 0; i < 3; ++i) {
      lapse_deriv_term.get(i) =
          (lapse_times_conformal_factor_flux.get(i) /
               get(lapse_times_conformal_factor) -
           7. * conformal_factor_flux.get(i) / get(conformal_factor));
    }
    auto& lapse_deriv_term_lo = get<::Tags::Tempi<2, 3>>(buffer);
    if constexpr (ConformalGeometry == Geometry::FlatCartesian) {
      get<0>(lapse_deriv_term_lo) = get<0>(lapse_deriv_term);
      get<1>(lapse_deriv_term_lo) = get<1>(lapse_deriv_term);
      get<2>(lapse_deriv_term_lo) = get<2>(lapse_deriv_term);
    } else {
      raise_or_lower_index(make_not_null(&lapse_deriv_term_lo),
                           lapse_deriv_term, conformal_metric->get());
    }
    for (size_t i = 0; i < 3; ++i) {
      // Add momentum density term
      momentum_constraint->get(i) +=
          16. * M_PI * get(lapse_times_conformal_factor) *
          pow<3 - ConformalMatterScale>(get(conformal_factor)) *
          conformal_momentum_density.get(i);
      // Add longitudinal shift term
      for (size_t j = 0; j < 3; ++j) {
        momentum_constraint->get(i) +=
            longitudinal_shift_minus_dt_conformal_metric.get(i, j) *
            lapse_deriv_term_lo.get(j);
      }
      momentum_constraint->get(i) -= minus_div_dt_conformal_metric.get(i);
    }
  }
}

template <int ConformalMatterScale>
void add_curved_momentum_sources(
    const gsl::not_null<Scalar<DataVector>*> hamiltonian_constraint,
    const gsl::not_null<Scalar<DataVector>*> lapse_equation,
    const gsl::not_null<tnsr::I<DataVector, 3>*> momentum_constraint,
    const tnsr::I<DataVector, 3>& conformal_momentum_density,
    const tnsr::i<DataVector, 3>& extrinsic_curvature_trace_gradient,
    const tnsr::ii<DataVector, 3>& conformal_metric,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::I<DataVector, 3>& minus_div_dt_conformal_metric,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const tnsr::I<DataVector, 3>& conformal_factor_flux,
    const tnsr::I<DataVector, 3>& lapse_times_conformal_factor_flux,
    const tnsr::II<DataVector, 3>&
        longitudinal_shift_minus_dt_conformal_metric) noexcept {
  add_momentum_sources_impl<ConformalMatterScale, Geometry::Curved>(
      hamiltonian_constraint, lapse_equation, momentum_constraint,
      conformal_momentum_density, extrinsic_curvature_trace_gradient,
      conformal_metric, inv_conformal_metric, minus_div_dt_conformal_metric,
      conformal_factor, lapse_times_conformal_factor, conformal_factor_flux,
      lapse_times_conformal_factor_flux,
      longitudinal_shift_minus_dt_conformal_metric);
}

template <int ConformalMatterScale>
void add_flat_cartesian_momentum_sources(
    const gsl::not_null<Scalar<DataVector>*> hamiltonian_constraint,
    const gsl::not_null<Scalar<DataVector>*> lapse_equation,
    const gsl::not_null<tnsr::I<DataVector, 3>*> momentum_constraint,
    const tnsr::I<DataVector, 3>& conformal_momentum_density,
    const tnsr::i<DataVector, 3>& extrinsic_curvature_trace_gradient,
    const tnsr::I<DataVector, 3>& minus_div_dt_conformal_metric,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const tnsr::I<DataVector, 3>& conformal_factor_flux,
    const tnsr::I<DataVector, 3>& lapse_times_conformal_factor_flux,
    const tnsr::II<DataVector, 3>&
        longitudinal_shift_minus_dt_conformal_metric) noexcept {
  add_momentum_sources_impl<ConformalMatterScale, Geometry::FlatCartesian>(
      hamiltonian_constraint, lapse_equation, momentum_constraint,
      conformal_momentum_density, extrinsic_curvature_trace_gradient,
      std::nullopt, std::nullopt, minus_div_dt_conformal_metric,
      conformal_factor, lapse_times_conformal_factor, conformal_factor_flux,
      lapse_times_conformal_factor_flux,
      longitudinal_shift_minus_dt_conformal_metric);
}

template <int ConformalMatterScale, Geometry ConformalGeometry>
void add_linearized_momentum_sources_impl(
    const gsl::not_null<Scalar<DataVector>*> linearized_hamiltonian_constraint,
    const gsl::not_null<Scalar<DataVector>*> linearized_lapse_equation,
    const gsl::not_null<tnsr::I<DataVector, 3>*> linearized_momentum_constraint,
    const tnsr::I<DataVector, 3>& conformal_momentum_density,
    const tnsr::i<DataVector, 3>& extrinsic_curvature_trace_gradient,
    const std::optional<std::reference_wrapper<const tnsr::ii<DataVector, 3>>>
        conformal_metric,
    const std::optional<std::reference_wrapper<const tnsr::II<DataVector, 3>>>
        inv_conformal_metric,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const tnsr::I<DataVector, 3>& conformal_factor_flux,
    const tnsr::I<DataVector, 3>& lapse_times_conformal_factor_flux,
    const tnsr::II<DataVector, 3>& longitudinal_shift_minus_dt_conformal_metric,
    const Scalar<DataVector>& conformal_factor_correction,
    const Scalar<DataVector>& lapse_times_conformal_factor_correction,
    const tnsr::I<DataVector, 3>& shift_correction,
    const tnsr::I<DataVector, 3>& conformal_factor_flux_correction,
    const tnsr::I<DataVector, 3>& lapse_times_conformal_factor_flux_correction,
    const tnsr::II<DataVector, 3>& longitudinal_shift_correction) noexcept {
  if constexpr (ConformalGeometry == Geometry::FlatCartesian) {
    ASSERT(not conformal_metric.has_value() and
               not inv_conformal_metric.has_value(),
           "You don't need to pass a conformal metric to this function when it "
           "is specialized for a flat conformal geometry in Cartesian "
           "coordinates.");
  } else {
    ASSERT(conformal_metric.has_value() and inv_conformal_metric.has_value(),
           "You must pass a conformal metric to this function when it is "
           "specialized for a curved conformal geometry.");
  }
  const size_t num_points = get(conformal_factor).size();
  TempBuffer<tmpl::list<::Tags::TempScalar<0>, ::Tags::TempI<1, 3>,
                        ::Tags::Tempi<2, 3>>>
      buffer{num_points};
  {
    // Add shift terms to linearized Hamiltonian and lapse equations
    auto& longitudinal_shift_square = get<::Tags::TempScalar<0>>(buffer);
    if constexpr (ConformalGeometry == Geometry::FlatCartesian) {
      detail::fully_contract_flat_cartesian(
          make_not_null(&longitudinal_shift_square),
          longitudinal_shift_minus_dt_conformal_metric,
          longitudinal_shift_minus_dt_conformal_metric);
    } else {
      auto& buffer1 = get<0>(get<::Tags::TempI<1, 3>>(buffer));
      auto& buffer2 = get<1>(get<::Tags::TempI<1, 3>>(buffer));
      detail::fully_contract(make_not_null(&longitudinal_shift_square),
                             make_not_null(&buffer1), make_not_null(&buffer2),
                             longitudinal_shift_minus_dt_conformal_metric,
                             longitudinal_shift_minus_dt_conformal_metric,
                             conformal_metric->get());
    }
    // Note: 0.21875 = 7 / 32, 0.0625 = 1 / 16 and 1.3125 = 21 / 16
    get(*linearized_hamiltonian_constraint) +=
        -0.21875 * pow<6>(get(conformal_factor)) /
            square(get(lapse_times_conformal_factor)) *
            get(longitudinal_shift_square) * get(conformal_factor_correction) +
        0.0625 * pow<7>(get(conformal_factor)) /
            pow<3>(get(lapse_times_conformal_factor)) *
            get(longitudinal_shift_square) *
            get(lapse_times_conformal_factor_correction);
    get(*linearized_lapse_equation) +=
        1.3125 * pow<5>(get(conformal_factor)) /
            get(lapse_times_conformal_factor) * get(longitudinal_shift_square) *
            get(conformal_factor_correction) -
        0.21875 * pow<6>(get(conformal_factor)) /
            square(get(lapse_times_conformal_factor)) *
            get(longitudinal_shift_square) *
            get(lapse_times_conformal_factor_correction);
  }
  {
    // Add shift-correction terms to linearized Hamiltonian and lapse equations,
    // re-using the buffer used for `longitudinal_shift_square` above
    auto& longitudinal_shift_dot_correction =
        get<::Tags::TempScalar<0>>(buffer);
    if constexpr (ConformalGeometry == Geometry::FlatCartesian) {
      detail::fully_contract_flat_cartesian(
          make_not_null(&longitudinal_shift_dot_correction),
          longitudinal_shift_minus_dt_conformal_metric,
          longitudinal_shift_correction);
    } else {
      auto& buffer1 = get<0>(get<::Tags::TempI<1, 3>>(buffer));
      auto& buffer2 = get<1>(get<::Tags::TempI<1, 3>>(buffer));
      detail::fully_contract(make_not_null(&longitudinal_shift_dot_correction),
                             make_not_null(&buffer1), make_not_null(&buffer2),
                             longitudinal_shift_minus_dt_conformal_metric,
                             longitudinal_shift_correction,
                             conformal_metric->get());
    }
    // Note: 0.0625 = 1 / 16 and 0.4375 = 7 / 16
    get(*linearized_hamiltonian_constraint) -=
        0.0625 * pow<7>(get(conformal_factor)) /
        square(get(lapse_times_conformal_factor)) *
        get(longitudinal_shift_dot_correction);
    get(*linearized_lapse_equation) += 0.4375 * pow<6>(get(conformal_factor)) /
                                       get(lapse_times_conformal_factor) *
                                       get(longitudinal_shift_dot_correction);
  }
  {
    // Add remaining linearization w.r.t. shift of the shift.grad(K) term to the
    // lapse equation. The linearization w.r.t. conformal factor and lapse is
    // already added in `linearized_lapse_sources`.
    auto& shift_correction_dot_extrinsic_curvature_trace_gradient =
        get<::Tags::TempScalar<0>>(buffer);
    dot_product(
        make_not_null(&shift_correction_dot_extrinsic_curvature_trace_gradient),
        shift_correction, extrinsic_curvature_trace_gradient);
    get(*linearized_lapse_equation) +=
        pow<5>(get(conformal_factor)) *
        get(shift_correction_dot_extrinsic_curvature_trace_gradient);
  }
  // Add sources to linearized momentum constraint
  {
    auto& extrinsic_curvature_trace_gradient_term =
        get<::Tags::TempI<1, 3>>(buffer);
    // Extrinsic curvature term
    if constexpr (ConformalGeometry == Geometry::FlatCartesian) {
      get<0>(extrinsic_curvature_trace_gradient_term) =
          get<0>(extrinsic_curvature_trace_gradient);
      get<1>(extrinsic_curvature_trace_gradient_term) =
          get<1>(extrinsic_curvature_trace_gradient);
      get<2>(extrinsic_curvature_trace_gradient_term) =
          get<2>(extrinsic_curvature_trace_gradient);
    } else {
      raise_or_lower_index(
          make_not_null(&extrinsic_curvature_trace_gradient_term),
          extrinsic_curvature_trace_gradient, inv_conformal_metric->get());
    }
    for (size_t i = 0; i < 3; ++i) {
      extrinsic_curvature_trace_gradient_term.get(i) *=
          4. / 3. *
          (get(lapse_times_conformal_factor_correction) /
               get(conformal_factor) -
           get(lapse_times_conformal_factor) / square(get(conformal_factor)) *
               get(conformal_factor_correction));
      linearized_momentum_constraint->get(i) +=
          extrinsic_curvature_trace_gradient_term.get(i);
    }
  }
  {
    // Compute lapse deriv term to be contracted with longitudinal
    // shift-correction
    auto& lapse_deriv_term = get<::Tags::TempI<1, 3>>(buffer);
    for (size_t i = 0; i < 3; ++i) {
      lapse_deriv_term.get(i) =
          (lapse_times_conformal_factor_flux.get(i) /
               get(lapse_times_conformal_factor) -
           7. * conformal_factor_flux.get(i) / get(conformal_factor));
    }
    auto& lapse_deriv_term_lo = get<::Tags::Tempi<2, 3>>(buffer);
    if constexpr (ConformalGeometry == Geometry::FlatCartesian) {
      get<0>(lapse_deriv_term_lo) = get<0>(lapse_deriv_term);
      get<1>(lapse_deriv_term_lo) = get<1>(lapse_deriv_term);
      get<2>(lapse_deriv_term_lo) = get<2>(lapse_deriv_term);
    } else {
      raise_or_lower_index(make_not_null(&lapse_deriv_term_lo),
                           lapse_deriv_term, conformal_metric->get());
    }
    for (size_t i = 0; i < 3; ++i) {
      // Add momentum density term
      linearized_momentum_constraint->get(i) +=
          16. * M_PI *
          (pow<3 - ConformalMatterScale>(get(conformal_factor)) *
               get(lapse_times_conformal_factor_correction) +
           (3. - ConformalMatterScale) *
               pow<2 - ConformalMatterScale>(get(conformal_factor)) *
               get(lapse_times_conformal_factor) *
               get(conformal_factor_correction)) *
          conformal_momentum_density.get(i);
      // Add longitudinal shift-correction term
      for (size_t j = 0; j < 3; ++j) {
        linearized_momentum_constraint->get(i) +=
            longitudinal_shift_correction.get(i, j) *
            lapse_deriv_term_lo.get(j);
      }
    }
  }
  {
    // Compute lapse deriv correction term to be contracted with longitudinal
    // shift
    auto& lapse_deriv_correction_term = get<::Tags::TempI<1, 3>>(buffer);
    for (size_t i = 0; i < 3; ++i) {
      lapse_deriv_correction_term.get(i) =
          (lapse_times_conformal_factor_flux_correction.get(i) /
               get(lapse_times_conformal_factor) -
           lapse_times_conformal_factor_flux.get(i) /
               square(get(lapse_times_conformal_factor)) *
               get(lapse_times_conformal_factor_correction) -
           7. * conformal_factor_flux_correction.get(i) /
               get(conformal_factor) +
           7. * conformal_factor_flux.get(i) / square(get(conformal_factor)) *
               get(conformal_factor_correction));
    }
    auto& lapse_deriv_correction_term_lo = get<::Tags::Tempi<2, 3>>(buffer);
    if constexpr (ConformalGeometry == Geometry::FlatCartesian) {
      get<0>(lapse_deriv_correction_term_lo) =
          get<0>(lapse_deriv_correction_term);
      get<1>(lapse_deriv_correction_term_lo) =
          get<1>(lapse_deriv_correction_term);
      get<2>(lapse_deriv_correction_term_lo) =
          get<2>(lapse_deriv_correction_term);
    } else {
      raise_or_lower_index(make_not_null(&lapse_deriv_correction_term_lo),
                           lapse_deriv_correction_term,
                           conformal_metric->get());
    }
    for (size_t i = 0; i < 3; ++i) {
      // Add longitudinal shift term
      for (size_t j = 0; j < 3; ++j) {
        linearized_momentum_constraint->get(i) +=
            longitudinal_shift_minus_dt_conformal_metric.get(i, j) *
            lapse_deriv_correction_term_lo.get(j);
      }
    }
  }
}

template <int ConformalMatterScale>
void add_curved_linearized_momentum_sources(
    gsl::not_null<Scalar<DataVector>*> linearized_hamiltonian_constraint,
    gsl::not_null<Scalar<DataVector>*> linearized_lapse_equation,
    gsl::not_null<tnsr::I<DataVector, 3>*> linearized_momentum_constraint,
    const tnsr::I<DataVector, 3>& conformal_momentum_density,
    const tnsr::i<DataVector, 3>& extrinsic_curvature_trace_gradient,
    const tnsr::ii<DataVector, 3>& conformal_metric,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const tnsr::I<DataVector, 3>& conformal_factor_flux,
    const tnsr::I<DataVector, 3>& lapse_times_conformal_factor_flux,
    const tnsr::II<DataVector, 3>& longitudinal_shift_minus_dt_conformal_metric,
    const Scalar<DataVector>& conformal_factor_correction,
    const Scalar<DataVector>& lapse_times_conformal_factor_correction,
    const tnsr::I<DataVector, 3>& shift_correction,
    const tnsr::I<DataVector, 3>& conformal_factor_flux_correction,
    const tnsr::I<DataVector, 3>& lapse_times_conformal_factor_flux_correction,
    const tnsr::II<DataVector, 3>& longitudinal_shift_correction) noexcept {
  add_linearized_momentum_sources_impl<ConformalMatterScale, Geometry::Curved>(
      linearized_hamiltonian_constraint, linearized_lapse_equation,
      linearized_momentum_constraint, conformal_momentum_density,
      extrinsic_curvature_trace_gradient, conformal_metric,
      inv_conformal_metric, conformal_factor, lapse_times_conformal_factor,
      conformal_factor_flux, lapse_times_conformal_factor_flux,
      longitudinal_shift_minus_dt_conformal_metric, conformal_factor_correction,
      lapse_times_conformal_factor_correction, shift_correction,
      conformal_factor_flux_correction,
      lapse_times_conformal_factor_flux_correction,
      longitudinal_shift_correction);
}

template <int ConformalMatterScale>
void add_flat_cartesian_linearized_momentum_sources(
    gsl::not_null<Scalar<DataVector>*> linearized_hamiltonian_constraint,
    gsl::not_null<Scalar<DataVector>*> linearized_lapse_equation,
    gsl::not_null<tnsr::I<DataVector, 3>*> linearized_momentum_constraint,
    const tnsr::I<DataVector, 3>& conformal_momentum_density,
    const tnsr::i<DataVector, 3>& extrinsic_curvature_trace_gradient,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const tnsr::I<DataVector, 3>& conformal_factor_flux,
    const tnsr::I<DataVector, 3>& lapse_times_conformal_factor_flux,
    const tnsr::II<DataVector, 3>& longitudinal_shift_minus_dt_conformal_metric,
    const Scalar<DataVector>& conformal_factor_correction,
    const Scalar<DataVector>& lapse_times_conformal_factor_correction,
    const tnsr::I<DataVector, 3>& shift_correction,
    const tnsr::I<DataVector, 3>& conformal_factor_flux_correction,
    const tnsr::I<DataVector, 3>& lapse_times_conformal_factor_flux_correction,
    const tnsr::II<DataVector, 3>& longitudinal_shift_correction) noexcept {
  add_linearized_momentum_sources_impl<ConformalMatterScale,
                                       Geometry::FlatCartesian>(
      linearized_hamiltonian_constraint, linearized_lapse_equation,
      linearized_momentum_constraint, conformal_momentum_density,
      extrinsic_curvature_trace_gradient, std::nullopt, std::nullopt,
      conformal_factor, lapse_times_conformal_factor, conformal_factor_flux,
      lapse_times_conformal_factor_flux,
      longitudinal_shift_minus_dt_conformal_metric, conformal_factor_correction,
      lapse_times_conformal_factor_correction, shift_correction,
      conformal_factor_flux_correction,
      lapse_times_conformal_factor_flux_correction,
      longitudinal_shift_correction);
}

// Instantiate all function templates

#define CONF_MATTER_SCALE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                                \
  template void add_hamiltonian_sources<CONF_MATTER_SCALE(data)>(             \
      gsl::not_null<Scalar<DataVector>*> hamiltonian_constraint,              \
      const Scalar<DataVector>& conformal_energy_density,                     \
      const Scalar<DataVector>& extrinsic_curvature_trace,                    \
      const Scalar<DataVector>& conformal_factor) noexcept;                   \
  template void add_linearized_hamiltonian_sources<CONF_MATTER_SCALE(data)>(  \
      gsl::not_null<Scalar<DataVector>*> linearized_hamiltonian_constraint,   \
      const Scalar<DataVector>& conformal_energy_density,                     \
      const Scalar<DataVector>& extrinsic_curvature_trace,                    \
      const Scalar<DataVector>& conformal_factor,                             \
      const Scalar<DataVector>& conformal_factor_correction) noexcept;        \
  template void add_lapse_sources<CONF_MATTER_SCALE(data)>(                   \
      gsl::not_null<Scalar<DataVector>*> lapse_equation,                      \
      const Scalar<DataVector>& conformal_energy_density,                     \
      const Scalar<DataVector>& conformal_stress_trace,                       \
      const Scalar<DataVector>& extrinsic_curvature_trace,                    \
      const Scalar<DataVector>& dt_extrinsic_curvature_trace,                 \
      const Scalar<DataVector>& shift_dot_deriv_extrinsic_curvature_trace,    \
      const Scalar<DataVector>& conformal_factor,                             \
      const Scalar<DataVector>& lapse_times_conformal_factor) noexcept;       \
  template void add_linearized_lapse_sources<CONF_MATTER_SCALE(data)>(        \
      gsl::not_null<Scalar<DataVector>*> linearized_lapse_equation,           \
      const Scalar<DataVector>& conformal_energy_density,                     \
      const Scalar<DataVector>& conformal_stress_trace,                       \
      const Scalar<DataVector>& extrinsic_curvature_trace,                    \
      const Scalar<DataVector>& dt_extrinsic_curvature_trace,                 \
      const Scalar<DataVector>& shift_dot_deriv_extrinsic_curvature_trace,    \
      const Scalar<DataVector>& conformal_factor,                             \
      const Scalar<DataVector>& lapse_times_conformal_factor,                 \
      const Scalar<DataVector>& conformal_factor_correction,                  \
      const Scalar<DataVector>&                                               \
          lapse_times_conformal_factor_correction) noexcept;                  \
  template void add_curved_momentum_sources<CONF_MATTER_SCALE(data)>(         \
      gsl::not_null<Scalar<DataVector>*> hamiltonian_constraint,              \
      gsl::not_null<Scalar<DataVector>*> lapse_equation,                      \
      gsl::not_null<tnsr::I<DataVector, 3>*> momentum_constraint,             \
      const tnsr::I<DataVector, 3>& conformal_momentum_density,               \
      const tnsr::i<DataVector, 3>& extrinsic_curvature_trace_gradient,       \
      const tnsr::ii<DataVector, 3>& conformal_metric,                        \
      const tnsr::II<DataVector, 3>& inv_conformal_metric,                    \
      const tnsr::I<DataVector, 3>& minus_div_dt_conformal_metric,            \
      const Scalar<DataVector>& conformal_factor,                             \
      const Scalar<DataVector>& lapse_times_conformal_factor,                 \
      const tnsr::I<DataVector, 3>& conformal_factor_flux,                    \
      const tnsr::I<DataVector, 3>& lapse_times_conformal_factor_flux,        \
      const tnsr::II<DataVector, 3>&                                          \
          longitudinal_shift_minus_dt_conformal_metric) noexcept;             \
  template void add_flat_cartesian_momentum_sources<CONF_MATTER_SCALE(data)>( \
      gsl::not_null<Scalar<DataVector>*> hamiltonian_constraint,              \
      gsl::not_null<Scalar<DataVector>*> lapse_equation,                      \
      gsl::not_null<tnsr::I<DataVector, 3>*> momentum_constraint,             \
      const tnsr::I<DataVector, 3>& conformal_momentum_density,               \
      const tnsr::i<DataVector, 3>& extrinsic_curvature_trace_gradient,       \
      const tnsr::I<DataVector, 3>& minus_div_dt_conformal_metric,            \
      const Scalar<DataVector>& conformal_factor,                             \
      const Scalar<DataVector>& lapse_times_conformal_factor,                 \
      const tnsr::I<DataVector, 3>& conformal_factor_flux,                    \
      const tnsr::I<DataVector, 3>& lapse_times_conformal_factor_flux,        \
      const tnsr::II<DataVector, 3>&                                          \
          longitudinal_shift_minus_dt_conformal_metric) noexcept;             \
  template void                                                               \
  add_curved_linearized_momentum_sources<CONF_MATTER_SCALE(data)>(            \
      gsl::not_null<Scalar<DataVector>*> linearized_hamiltonian_constraint,   \
      gsl::not_null<Scalar<DataVector>*> linearized_lapse_equation,           \
      gsl::not_null<tnsr::I<DataVector, 3>*> linearized_momentum_constraint,  \
      const tnsr::I<DataVector, 3>& conformal_momentum_density,               \
      const tnsr::i<DataVector, 3>& extrinsic_curvature_trace_gradient,       \
      const tnsr::ii<DataVector, 3>& conformal_metric,                        \
      const tnsr::II<DataVector, 3>& inv_conformal_metric,                    \
      const Scalar<DataVector>& conformal_factor,                             \
      const Scalar<DataVector>& lapse_times_conformal_factor,                 \
      const tnsr::I<DataVector, 3>& conformal_factor_flux,                    \
      const tnsr::I<DataVector, 3>& lapse_times_conformal_factor_flux,        \
      const tnsr::II<DataVector, 3>&                                          \
          longitudinal_shift_minus_dt_conformal_metric,                       \
      const Scalar<DataVector>& conformal_factor_correction,                  \
      const Scalar<DataVector>& lapse_times_conformal_factor_correction,      \
      const tnsr::I<DataVector, 3>& shift_correction,                         \
      const tnsr::I<DataVector, 3>& conformal_factor_flux_correction,         \
      const tnsr::I<DataVector, 3>&                                           \
          lapse_times_conformal_factor_flux_correction,                       \
      const tnsr::II<DataVector, 3>& longitudinal_shift_correction) noexcept; \
  template void                                                               \
  add_flat_cartesian_linearized_momentum_sources<CONF_MATTER_SCALE(data)>(    \
      gsl::not_null<Scalar<DataVector>*> linearized_hamiltonian_constraint,   \
      gsl::not_null<Scalar<DataVector>*> linearized_lapse_equation,           \
      gsl::not_null<tnsr::I<DataVector, 3>*> linearized_momentum_constraint,  \
      const tnsr::I<DataVector, 3>& conformal_momentum_density,               \
      const tnsr::i<DataVector, 3>& extrinsic_curvature_trace_gradient,       \
      const Scalar<DataVector>& conformal_factor,                             \
      const Scalar<DataVector>& lapse_times_conformal_factor,                 \
      const tnsr::I<DataVector, 3>& conformal_factor_flux,                    \
      const tnsr::I<DataVector, 3>& lapse_times_conformal_factor_flux,        \
      const tnsr::II<DataVector, 3>&                                          \
          longitudinal_shift_minus_dt_conformal_metric,                       \
      const Scalar<DataVector>& conformal_factor_correction,                  \
      const Scalar<DataVector>& lapse_times_conformal_factor_correction,      \
      const tnsr::I<DataVector, 3>& shift_correction,                         \
      const tnsr::I<DataVector, 3>& conformal_factor_flux_correction,         \
      const tnsr::I<DataVector, 3>&                                           \
          lapse_times_conformal_factor_flux_correction,                       \
      const tnsr::II<DataVector, 3>& longitudinal_shift_correction) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (0, 6, 8))

#undef CONF_MATTER_SCALE
#undef INSTANTIATION

}  // namespace Xcts
