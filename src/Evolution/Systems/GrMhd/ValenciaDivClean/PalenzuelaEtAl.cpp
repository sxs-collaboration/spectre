// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/PalenzuelaEtAl.hpp"

#include <boost/none.hpp>
#include <cmath>
#include <exception>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveRecoveryData.hpp"
#include "NumericalAlgorithms/RootFinding/TOMS748.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Overloader.hpp"

// IWYU pragma: no_forward_declare EquationsOfState::EquationOfState

/// \cond
namespace grmhd {
namespace ValenciaDivClean {
namespace PrimitiveRecoverySchemes {

namespace {

// note q,r,s,t,x are defined in the documentation
template <size_t ThermodynamicDim>
class FunctionOfX {
 public:
  FunctionOfX(const double total_energy_density,
              const double momentum_density_squared,
              const double momentum_density_dot_magnetic_field,
              const double magnetic_field_squared,
              const double rest_mass_density_times_lorentz_factor,
              const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
                  equation_of_state) noexcept
      : q_(total_energy_density / rest_mass_density_times_lorentz_factor - 1.0),
        r_(momentum_density_squared /
           square(rest_mass_density_times_lorentz_factor)),
        s_(magnetic_field_squared / rest_mass_density_times_lorentz_factor),
        t_squared_(square(momentum_density_dot_magnetic_field) /
                   cube(rest_mass_density_times_lorentz_factor)),
        rest_mass_density_times_lorentz_factor_(
            rest_mass_density_times_lorentz_factor),
        equation_of_state_(equation_of_state) {}

  double lorentz_factor(const double x) const noexcept {
    static constexpr double v_maximum = 1.0 - 1.e-12;
    // Clamp v^2 to physical values.  This is needed because the bounds on
    // x used for the root solve do not guarantee a physical velocity.  Some
    // work would be needed to investigate whether better bounds could guarantee
    // a physical velocity.
    const double v_squared = cpp17::clamp(
        (square(x) * r_ + (2 * x + s_) * t_squared_) / square(x * (x + s_)),
        0.0, square(v_maximum));
    return 1.0 / sqrt(1.0 - v_squared);
  }

  double specific_internal_energy(const double x,
                                  const double lorentz_factor) const noexcept {
    return lorentz_factor - 1.0 +
           x * (1.0 - square(lorentz_factor)) / lorentz_factor +
           lorentz_factor * (q_ - s_ + 0.5 * t_squared_ / square(x) +
                             0.5 * s_ / square(lorentz_factor));
  }

  double operator()(const double x) const noexcept {
    const double current_lorentz_factor = lorentz_factor(x);
    const double current_rest_mass_density =
        rest_mass_density_times_lorentz_factor_ / current_lorentz_factor;
    const double current_specific_internal_energy =
        specific_internal_energy(x, current_lorentz_factor);
    const double current_pressure = get(make_overloader(
        [&current_rest_mass_density](
            const EquationsOfState::EquationOfState<true, 1>&
                the_equation_of_state) noexcept {
          return the_equation_of_state.pressure_from_density(
              Scalar<double>(current_rest_mass_density));
        },
        [&current_rest_mass_density, &current_specific_internal_energy ](
            const EquationsOfState::EquationOfState<true, 2>&
                the_equation_of_state) noexcept {
          return the_equation_of_state.pressure_from_density_and_energy(
              Scalar<double>(current_rest_mass_density),
              Scalar<double>(current_specific_internal_energy));
        })(equation_of_state_));

    return x - (1.0 + current_specific_internal_energy +
                current_pressure / current_rest_mass_density) *
                   current_lorentz_factor;
  }

 private:
  const double q_;
  const double r_;
  const double s_;
  const double t_squared_;
  const double rest_mass_density_times_lorentz_factor_;
  const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
      equation_of_state_;
};
}  // namespace

template <size_t ThermodynamicDim>
boost::optional<PrimitiveRecoveryData> PalenzuelaEtAl::apply(
    const double /*initial_guess_pressure*/, const double total_energy_density,
    const double momentum_density_squared,
    const double momentum_density_dot_magnetic_field,
    const double magnetic_field_squared,
    const double rest_mass_density_times_lorentz_factor,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) noexcept {
  const double lower_bound = (total_energy_density - magnetic_field_squared) /
                             rest_mass_density_times_lorentz_factor;
  const double upper_bound =
      (2.0 * total_energy_density - magnetic_field_squared) /
      rest_mass_density_times_lorentz_factor;
  const auto f_of_x =
      FunctionOfX<ThermodynamicDim>{total_energy_density,
                                    momentum_density_squared,
                                    momentum_density_dot_magnetic_field,
                                    magnetic_field_squared,
                                    rest_mass_density_times_lorentz_factor,
                                    equation_of_state};
  double specific_enthalpy_times_lorentz_factor;
  try {
    specific_enthalpy_times_lorentz_factor =
        // NOLINTNEXTLINE(clang-analyzer-core)
        RootFinder::toms748(f_of_x, lower_bound, upper_bound,
                            absolute_tolerance_, relative_tolerance_,
                            max_iterations_);
  } catch (std::exception& exception) {
    return boost::none;
  }
  const double lorentz_factor =
      f_of_x.lorentz_factor(specific_enthalpy_times_lorentz_factor);
  const double rest_mass_density =
      rest_mass_density_times_lorentz_factor / lorentz_factor;
  const double specific_internal_energy = f_of_x.specific_internal_energy(
      specific_enthalpy_times_lorentz_factor, lorentz_factor);
  const double pressure = get(make_overloader(
      [&rest_mass_density](const EquationsOfState::EquationOfState<true, 1>&
                               the_equation_of_state) noexcept {
        return the_equation_of_state.pressure_from_density(
            Scalar<double>(rest_mass_density));
      },
      [&rest_mass_density, &specific_internal_energy ](
          const EquationsOfState::EquationOfState<true, 2>&
              the_equation_of_state) noexcept {
        return the_equation_of_state.pressure_from_density_and_energy(
            Scalar<double>(rest_mass_density),
            Scalar<double>(specific_internal_energy));
      })(equation_of_state));

  return PrimitiveRecoveryData{rest_mass_density, lorentz_factor, pressure,
                               specific_enthalpy_times_lorentz_factor *
                                   rest_mass_density_times_lorentz_factor};
}
}  // namespace PrimitiveRecoverySchemes
}  // namespace ValenciaDivClean
}  // namespace grmhd

#define THERMODIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATION(_, data)                                                 \
  template boost::optional<grmhd::ValenciaDivClean::PrimitiveRecoverySchemes:: \
                               PrimitiveRecoveryData>                          \
  grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::PalenzuelaEtAl::apply<    \
      THERMODIM(data)>(                                                        \
      const double /*initial_guess_pressure*/,                                 \
      const double total_energy_density,                                       \
      const double momentum_density_squared,                                   \
      const double momentum_density_dot_magnetic_field,                        \
      const double magnetic_field_squared,                                     \
      const double rest_mass_density_times_lorentz_factor,                     \
      const EquationsOfState::EquationOfState<true, THERMODIM(data)>&          \
          equation_of_state) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2))

#undef INSTANTIATION
#undef THERMODIM
/// \endcond
