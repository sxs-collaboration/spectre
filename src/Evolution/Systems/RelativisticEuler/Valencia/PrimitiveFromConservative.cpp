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
#include "Utilities/Overloader.hpp"

// IWYU pragma: no_forward_declare EquationsOfState::EquationOfState

/// \cond
namespace RelativisticEuler {
namespace Valencia {

namespace {

template <typename DataType, size_t ThermodynamicDim>
class FunctionOfZ {
 public:
  FunctionOfZ(const Scalar<DataType>& tilde_d,
              const Scalar<DataType>& tilde_tau,
              const Scalar<DataType>& tilde_s_magnitude,
              const Scalar<DataType>& sqrt_det_spatial_metric,
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

    const double pressure =
        make_overloader(
            [&rho](const EquationsOfState::EquationOfState<true, 1>&
                       equation_of_state) noexcept {
              return equation_of_state.pressure_from_density(
                  Scalar<double>(rho));
            },
            [&rho, &epsilon ](const EquationsOfState::EquationOfState<true, 2>&
                                  equation_of_state) noexcept {
              return equation_of_state.pressure_from_density_and_energy(
                  Scalar<double>(rho), Scalar<double>(epsilon));
            })(equation_of_state_)
            .get();

    const double a = pressure / e;
    const double h = (1.0 + epsilon) * (1.0 + a);
    return z - r / h;
  }

 private:
  const Scalar<DataType>& tilde_d_;
  const Scalar<DataType>& tilde_tau_;
  const Scalar<DataType>& tilde_s_magnitude_;
  const Scalar<DataType>& sqrt_det_spatial_metric_;
  const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
      equation_of_state_;
};
}  // namespace

template <size_t ThermodynamicDim, typename DataType, size_t Dim>
void primitive_from_conservative(
    const gsl::not_null<Scalar<DataType>*> rest_mass_density,
    const gsl::not_null<Scalar<DataType>*> specific_internal_energy,
    const gsl::not_null<Scalar<DataType>*> lorentz_factor,
    const gsl::not_null<Scalar<DataType>*> specific_enthalpy,
    const gsl::not_null<Scalar<DataType>*> pressure,
    const gsl::not_null<tnsr::I<DataType, Dim, Frame::Inertial>*>
        spatial_velocity,
    const Scalar<DataType>& tilde_d, const Scalar<DataType>& tilde_tau,
    const tnsr::i<DataType, Dim, Frame::Inertial>& tilde_s,
    const tnsr::II<DataType, Dim, Frame::Inertial>& inv_spatial_metric,
    const Scalar<DataType>& sqrt_det_spatial_metric,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) noexcept {
  const auto tilde_s_M = raise_or_lower_index(tilde_s, inv_spatial_metric);
  const Scalar<DataType> tilde_s_magnitude{
      sqrt(get(dot_product(tilde_s, tilde_s_M)))};

  // find z via root find
  // k not const to save allocation later
  DataType k = get(tilde_s_magnitude) / (get(tilde_tau) + get(tilde_d));
  const DataType lower_bound = 0.5 * k / sqrt(1.0 - 0.25 * square(k));
  const DataType upper_bound = k / sqrt(1.0 - square(k));
  const auto f_of_z = FunctionOfZ<DataType, ThermodynamicDim>{
      tilde_d, tilde_tau, tilde_s_magnitude, sqrt_det_spatial_metric,
      equation_of_state};

  const auto z =
      // NOLINTNEXTLINE(clang-analyzer-core)
      RootFinder::toms748(f_of_z, lower_bound, upper_bound,
                          10.0 * std::numeric_limits<double>::epsilon(),
                          10.0 * std::numeric_limits<double>::epsilon(), 100);
  get(*lorentz_factor) = sqrt(1.0 + square(z));
  get(*rest_mass_density) =
      get(tilde_d) / (get(*lorentz_factor) * get(sqrt_det_spatial_metric));
  get(*specific_internal_energy) =
      (get(*lorentz_factor) * get(tilde_tau) - z * get(tilde_s_magnitude)) /
          get(tilde_d) +
      square(z) / (1.0 + get(*lorentz_factor));

  *pressure = make_overloader(
      [&rest_mass_density](const EquationsOfState::EquationOfState<true, 1>&
                               the_equation_of_state) noexcept {
        return the_equation_of_state.pressure_from_density(*rest_mass_density);
      },
      [&rest_mass_density, &specific_internal_energy ](
          const EquationsOfState::EquationOfState<true, 2>&
              the_equation_of_state) noexcept {
        return the_equation_of_state.pressure_from_density_and_energy(
            *rest_mass_density, *specific_internal_energy);
      })(equation_of_state);

  get(*specific_enthalpy) = 1.0 + get(*specific_internal_energy) +
                            get(*pressure) / get(*rest_mass_density);

  // reuse k as a temporary
  DataType denominator = std::move(k);
  denominator = get(tilde_d) * get(*lorentz_factor) * get(*specific_enthalpy);
  for (size_t d = 0; d < Dim; ++d) {
    spatial_velocity->get(d) = tilde_s_M.get(d) / denominator;
  }
}

}  // namespace Valencia
}  // namespace RelativisticEuler

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM(data) BOOST_PP_TUPLE_ELEM(1, data)
#define THERMODIM(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATION(_, data)                                               \
  template void                                                              \
  RelativisticEuler::Valencia::primitive_from_conservative<THERMODIM(data)>( \
      const gsl::not_null<Scalar<DTYPE(data)>*> rest_mass_density,           \
      const gsl::not_null<Scalar<DTYPE(data)>*> specific_internal_energy,    \
      const gsl::not_null<Scalar<DTYPE(data)>*> lorentz_factor,              \
      const gsl::not_null<Scalar<DTYPE(data)>*> specific_enthalpy,           \
      const gsl::not_null<Scalar<DTYPE(data)>*> pressure,                    \
      const gsl::not_null<tnsr::I<DTYPE(data), DIM(data), Frame::Inertial>*> \
          spatial_velocity,                                                  \
      const Scalar<DTYPE(data)>& tilde_d,                                    \
      const Scalar<DTYPE(data)>& tilde_tau,                                  \
      const tnsr::i<DTYPE(data), DIM(data), Frame::Inertial>& tilde_s,       \
      const tnsr::II<DTYPE(data), DIM(data), Frame::Inertial>&               \
          inv_spatial_metric,                                                \
      const Scalar<DTYPE(data)>& sqrt_det_spatial_metric,                    \
      const EquationsOfState::EquationOfState<true, THERMODIM(data)>&        \
          equation_of_state) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (double, DataVector), (1, 2, 3), (1, 2))

#undef INSTANTIATION
#undef THERMODIM
#undef DIM
#undef DTYPE
/// \endcond
