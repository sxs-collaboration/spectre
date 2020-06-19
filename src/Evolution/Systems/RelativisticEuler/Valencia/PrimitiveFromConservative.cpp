// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/RelativisticEuler/Valencia/PrimitiveFromConservative.hpp"

#include <cmath>
#include <limits>

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/RootFinding/TOMS748.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

// IWYU pragma: no_forward_declare EquationsOfState::EquationOfState

/// \cond
namespace RelativisticEuler {
namespace Valencia {

namespace {

template <size_t ThermodynamicDim>
class FunctionOfZ {
 public:
  FunctionOfZ(const Scalar<DataVector>& tilde_d,
              const Scalar<DataVector>& tilde_tau,
              const Scalar<DataVector>& tilde_s_magnitude,
              const Scalar<DataVector>& sqrt_det_spatial_metric,
              const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
                  equation_of_state) noexcept
      : tilde_d_(tilde_d),
        tilde_tau_(tilde_tau),
        tilde_s_magnitude_(tilde_s_magnitude),
        sqrt_det_spatial_metric_(sqrt_det_spatial_metric),
        equation_of_state_(equation_of_state) {}

  double operator()(const double z, const size_t i = 0) const noexcept {
    const double r =
        get_element(get(tilde_s_magnitude_), i) / get_element(get(tilde_d_), i);
    const double q =
        get_element(get(tilde_tau_), i) / get_element(get(tilde_d_), i);
    const double W = sqrt(1.0 + square(z));
    const double rho = get_element(get(tilde_d_), i) /
                       (W * get_element(get(sqrt_det_spatial_metric_), i));
    // Note z^2/(1+W) is numerically more accurate than W-1 for small
    // velocities.
    const double epsilon = W * q - z * r + square(z) / (1.0 + W);
    const double e = rho * (1.0 + epsilon);

    double pressure = std::numeric_limits<double>::signaling_NaN();
    if constexpr (ThermodynamicDim == 1) {
      pressure =
          get(equation_of_state_.pressure_from_density(Scalar<double>(rho)));
    } else if constexpr (ThermodynamicDim == 2) {
      pressure = get(equation_of_state_.pressure_from_density_and_energy(
          Scalar<double>(rho), Scalar<double>(epsilon)));
    }

    const double a = pressure / e;
    const double h = (1.0 + epsilon) * (1.0 + a);
    return z - r / h;
  }

 private:
  const Scalar<DataVector>& tilde_d_;
  const Scalar<DataVector>& tilde_tau_;
  const Scalar<DataVector>& tilde_s_magnitude_;
  const Scalar<DataVector>& sqrt_det_spatial_metric_;
  const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
      equation_of_state_;
};
}  // namespace

template <size_t ThermodynamicDim, size_t Dim>
void PrimitiveFromConservative<ThermodynamicDim, Dim>::apply(
    const gsl::not_null<Scalar<DataVector>*> rest_mass_density,
    const gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
    const gsl::not_null<Scalar<DataVector>*> lorentz_factor,
    const gsl::not_null<Scalar<DataVector>*> specific_enthalpy,
    const gsl::not_null<Scalar<DataVector>*> pressure,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        spatial_velocity,
    const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_tau,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& tilde_s,
    const tnsr::II<DataVector, Dim, Frame::Inertial>& inv_spatial_metric,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) noexcept {
  const auto tilde_s_M = raise_or_lower_index(tilde_s, inv_spatial_metric);
  const Scalar<DataVector> tilde_s_magnitude{
      sqrt(get(dot_product(tilde_s, tilde_s_M)))};

  // find z via root find
  // k not const to save allocation later
  DataVector k = get(tilde_s_magnitude) / (get(tilde_tau) + get(tilde_d));
  const DataVector lower_bound = 0.5 * k / sqrt(1.0 - 0.25 * square(k));
  const DataVector upper_bound = k / sqrt(1.0 - square(k));
  const auto f_of_z =
      FunctionOfZ<ThermodynamicDim>{tilde_d, tilde_tau, tilde_s_magnitude,
                                    sqrt_det_spatial_metric, equation_of_state};

  DataVector z;
  try {
    // NOLINTNEXTLINE(clang-analyzer-core)
    z = RootFinder::toms748(f_of_z, lower_bound, upper_bound,
                            10.0 * std::numeric_limits<double>::epsilon(),
                            10.0 * std::numeric_limits<double>::epsilon(), 100);
  } catch (std::exception& exception) {
    ERROR(
        "Failed to find the intermediate variable z with TOMS748 root finder "
        "while computing the primitive variables from the conserved variables. "
        "Got exception message:\n"
        << exception.what());
  }
  get(*lorentz_factor) = sqrt(1.0 + square(z));
  get(*rest_mass_density) =
      get(tilde_d) / (get(*lorentz_factor) * get(sqrt_det_spatial_metric));
  get(*specific_internal_energy) =
      (get(*lorentz_factor) * get(tilde_tau) - z * get(tilde_s_magnitude)) /
          get(tilde_d) +
      square(z) / (1.0 + get(*lorentz_factor));

  if constexpr (ThermodynamicDim == 1) {
    *pressure = equation_of_state.pressure_from_density(*rest_mass_density);
  } else if constexpr (ThermodynamicDim == 2) {
    *pressure = equation_of_state.pressure_from_density_and_energy(
        *rest_mass_density, *specific_internal_energy);
  }

  get(*specific_enthalpy) = 1.0 + get(*specific_internal_energy) +
                            get(*pressure) / get(*rest_mass_density);

  // reuse k as a temporary
  DataVector denominator = std::move(k);
  denominator = get(tilde_d) * get(*lorentz_factor) * get(*specific_enthalpy);
  for (size_t d = 0; d < Dim; ++d) {
    spatial_velocity->get(d) = tilde_s_M.get(d) / denominator;
  }
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define THERMODIM(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATION(_, data) \
  template class PrimitiveFromConservative<THERMODIM(data), DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3), (1, 2))

#undef INSTANTIATION
#undef THERMODIM
#undef DIM
}  // namespace Valencia
}  // namespace RelativisticEuler
/// \endcond
