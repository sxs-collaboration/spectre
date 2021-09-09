// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/Characteristics.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/RelativisticEuler/Valencia/Characteristics.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/Tags.hpp"              // IWYU pragma: keep
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

// IWYU pragma: no_forward_declare EquationsOfState::EquationOfState
// IWYU pragma: no_forward_declare Tensor

namespace grmhd::ValenciaDivClean::detail {
Matrix flux_jacobian(const tnsr::i<double, 3>& /*unit_normal*/) noexcept {
  // TODO: Implement this following the example of the NewtonianEuler case.
  // A starting point for deriving the matrix: https://arxiv.org/abs/1503.00978
  // But note that this reference does not have the div-cleaning field, so
  // we will need a 9x9 matrix here (and not 8x8 as in the reference).
  // We will also need a lot of additional function arguments.
  return Matrix(9, 9, 0.0);
}
}  // namespace grmhd::ValenciaDivClean::detail

namespace {
void compute_characteristic_speeds(
    const gsl::not_null<std::array<DataVector, 9>*> pchar_speeds,
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift,
    const tnsr::I<DataVector, 3>& spatial_velocity,
    const Scalar<DataVector>& spatial_velocity_squared,
    const Scalar<DataVector>& sound_speed_squared,
    const Scalar<DataVector>& alfven_speed_squared,
    const tnsr::i<DataVector, 3>& normal) noexcept {
  const size_t num_grid_points = get(lapse).size();
  auto& char_speeds = *pchar_speeds;
  if (char_speeds[0].size() != num_grid_points) {
    char_speeds[0] = DataVector(num_grid_points);
  }
  Scalar<DataVector> temp0(char_speeds[0].data(), num_grid_points);
  dot_product(make_not_null(&temp0), normal, shift);
  char_speeds[0] *= -1.0;
  char_speeds[8] = char_speeds[0] + get(lapse);

  char_speeds[0] -= get(lapse);
  // Mapping of indices between GRMHD char speeds and relativistic Euler char
  // speeds arrays.
  //
  // GRMHD     Rel Euler
  //   1           0
  //   2           1
  //   3           2
  //   4           3
  //   5           1
  //   6           1
  //   7           4
  //
  // Create an array of non-owning DataVectors to be passed to the Relativistic
  // Euler char speed computation as a not_null<array<DataVectors>>.
  std::array<DataVector, 5> rel_euler_char_speeds{};
  for (size_t i = 0; i < 4; ++i) {
    if (gsl::at(char_speeds, i + 1).size() != num_grid_points) {
      gsl::at(char_speeds, i + 1) = DataVector(num_grid_points);
    }
    gsl::at(rel_euler_char_speeds, i)
        .set_data_ref(&gsl::at(char_speeds, i + 1));
  }
  if (gsl::at(char_speeds, 7).size() != num_grid_points) {
    gsl::at(char_speeds, 7) = DataVector(num_grid_points);
  }
  rel_euler_char_speeds[4].set_data_ref(&(char_speeds[7]));

  RelativisticEuler::Valencia::characteristic_speeds(
      make_not_null(&rel_euler_char_speeds), lapse, shift, spatial_velocity,
      spatial_velocity_squared,
      Scalar<DataVector>{get(sound_speed_squared) +
                         get(alfven_speed_squared) *
                             (1.0 - get(sound_speed_squared))},
      normal);

  for (size_t i = 5; i < 7; ++i) {
    gsl::at(char_speeds, i) = rel_euler_char_speeds[1];
  }
}
}  // namespace

namespace grmhd::ValenciaDivClean {
template <size_t ThermodynamicDim>
void characteristic_speeds(
    const gsl::not_null<std::array<DataVector, 9>*> char_speeds,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& specific_enthalpy,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& lorentz_factor,
    const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::i<DataVector, 3>& unit_normal,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) noexcept {
  // Use a Variables to reduce total number of allocations. This is especially
  // important in a multithreaded environment.
  // Remaining places to reduce allocations:
  // - EoS calls: 2 allocations
  // - Pass temp pointer to Rel Euler: 1 allocation
  // - Return a DataVectorArray (not yet implemented): 9 allocations
  Variables<tmpl::list<hydro::Tags::SpatialVelocityOneForm<DataVector, 3>,
                       hydro::Tags::SpatialVelocitySquared<DataVector>,
                       hydro::Tags::MagneticFieldOneForm<DataVector, 3>,
                       hydro::Tags::MagneticFieldDotSpatialVelocity<DataVector>,
                       hydro::Tags::MagneticFieldSquared<DataVector>,
                       hydro::Tags::ComovingMagneticFieldSquared<DataVector>,
                       hydro::Tags::SoundSpeedSquared<DataVector>>>
      temp_tensors{get<0>(shift).size()};

  const auto& spatial_velocity_one_form =
      get<hydro::Tags::SpatialVelocityOneForm<DataVector, 3>>(temp_tensors);
  raise_or_lower_index(
      make_not_null(&get<hydro::Tags::SpatialVelocityOneForm<DataVector, 3>>(
          temp_tensors)),
      spatial_velocity, spatial_metric);
  const auto& magnetic_field_one_form =
      get<hydro::Tags::MagneticFieldOneForm<DataVector, 3>>(temp_tensors);
  raise_or_lower_index(
      make_not_null(
          &get<hydro::Tags::MagneticFieldOneForm<DataVector, 3>>(temp_tensors)),
      magnetic_field, spatial_metric);
  const auto& magnetic_field_dot_spatial_velocity =
      get<hydro::Tags::MagneticFieldDotSpatialVelocity<DataVector>>(
          temp_tensors);
  dot_product(
      make_not_null(
          &get<hydro::Tags::MagneticFieldDotSpatialVelocity<DataVector>>(
              temp_tensors)),
      magnetic_field, spatial_velocity_one_form);
  const auto& spatial_velocity_squared =
      get<hydro::Tags::SpatialVelocitySquared<DataVector>>(temp_tensors);
  dot_product(
      make_not_null(
          &get<hydro::Tags::SpatialVelocitySquared<DataVector>>(temp_tensors)),
      spatial_velocity, spatial_velocity_one_form);

  const auto& magnetic_field_squared =
      get<hydro::Tags::MagneticFieldSquared<DataVector>>(temp_tensors);
  dot_product(make_not_null(&get<hydro::Tags::MagneticFieldSquared<DataVector>>(
                  temp_tensors)),
              magnetic_field, magnetic_field_one_form);
  const auto& comoving_magnetic_field_squared =
      get<hydro::Tags::ComovingMagneticFieldSquared<DataVector>>(temp_tensors);
  get(get<hydro::Tags::ComovingMagneticFieldSquared<DataVector>>(
      temp_tensors)) =
      get(magnetic_field_squared) / square(get(lorentz_factor)) +
      square(get(magnetic_field_dot_spatial_velocity));

  // reuse magnetic_field_squared allocation for Alfven speed squared
  const auto& alfven_speed_squared =
      get<hydro::Tags::MagneticFieldSquared<DataVector>>(temp_tensors);
  get(get<hydro::Tags::MagneticFieldSquared<DataVector>>(temp_tensors)) =
      get(comoving_magnetic_field_squared) /
      (get(comoving_magnetic_field_squared) +
       get(rest_mass_density) * get(specific_enthalpy));

  Scalar<DataVector>& sound_speed_squared =
      get<hydro::Tags::SoundSpeedSquared<DataVector>>(temp_tensors);
  if constexpr (ThermodynamicDim == 1) {
    get(sound_speed_squared) =
        get(equation_of_state.chi_from_density(rest_mass_density)) +
        get(equation_of_state.kappa_times_p_over_rho_squared_from_density(
            rest_mass_density));
  } else if constexpr (ThermodynamicDim == 2) {
    get(sound_speed_squared) =
        get(equation_of_state.chi_from_density_and_energy(
            rest_mass_density, specific_internal_energy)) +
        get(equation_of_state
                .kappa_times_p_over_rho_squared_from_density_and_energy(
                    rest_mass_density, specific_internal_energy));
  }
  get(sound_speed_squared) /= get(specific_enthalpy);

  compute_characteristic_speeds(char_speeds, lapse, shift, spatial_velocity,
                                spatial_velocity_squared, sound_speed_squared,
                                alfven_speed_squared, unit_normal);
}

template <size_t ThermodynamicDim>
std::array<DataVector, 9> characteristic_speeds(
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& specific_enthalpy,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& lorentz_factor,
    const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::i<DataVector, 3>& unit_normal,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) noexcept {
  std::array<DataVector, 9> char_speeds{};
  characteristic_speeds(make_not_null(&char_speeds), rest_mass_density,
                        specific_internal_energy, specific_enthalpy,
                        spatial_velocity, lorentz_factor, magnetic_field, lapse,
                        shift, spatial_metric, unit_normal, equation_of_state);
  return char_speeds;
}

template <size_t ThermodynamicDim>
std::pair<DataVector, std::pair<Matrix, Matrix>> numerical_eigensystem(
    const Scalar<double>& rest_mass_density,
    const Scalar<double>& specific_internal_energy,
    const Scalar<double>& specific_enthalpy,
    const tnsr::I<double, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<double>& lorentz_factor,
    const tnsr::I<double, 3, Frame::Inertial>& magnetic_field,
    const Scalar<double>& lapse, const tnsr::I<double, 3>& shift,
    const tnsr::ii<double, 3, Frame::Inertial>& spatial_metric,
    const tnsr::i<double, 3>& unit_normal,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) noexcept {
  ASSERT(equal_within_roundoff(get(magnitude(unit_normal)), 1.),
         "Expected unit normal, but got normal with magnitude "
             << get(magnitude(unit_normal)));

  // TODO: Fill in arguments for this call. May require adding new arguments to
  // the numerical_eigensystem function.
  const Matrix a = detail::flux_jacobian(unit_normal);

  // TODO: The code block below is a hack to call the characteristic_speeds
  // function with Tensor<DataVector> types when here we are working instead
  // with Tensor<double> types (because this funtion is designed for
  // cell-average char transforms for limiter testing).
  // A better solution may be to template characteristic_speeds on DataType?
  Scalar<DataVector> dv_rest_mass_density;
  Scalar<DataVector> dv_specific_internal_energy;
  Scalar<DataVector> dv_specific_enthalpy;
  tnsr::I<DataVector, 3> dv_spatial_velocity;
  Scalar<DataVector> dv_lorentz_factor;
  tnsr::I<DataVector, 3> dv_magnetic_field;
  Scalar<DataVector> dv_lapse;
  tnsr::I<DataVector, 3> dv_shift;
  tnsr::ii<DataVector, 3> dv_spatial_metric;
  tnsr::i<DataVector, 3> dv_unit_normal;
  get(dv_rest_mass_density) = DataVector(1, get(rest_mass_density));
  get(dv_specific_internal_energy) =
      DataVector(1, get(specific_internal_energy));
  get(dv_specific_enthalpy) = DataVector(1, get(specific_enthalpy));
  for (size_t i = 0; i < 3; ++i) {
    dv_spatial_velocity.get(i) = DataVector(1, spatial_velocity.get(i));
  }
  get(dv_lorentz_factor) = DataVector(1, get(lorentz_factor));
  for (size_t i = 0; i < 3; ++i) {
    dv_magnetic_field.get(i) = DataVector(1, magnetic_field.get(i));
  }
  get(dv_lapse) = DataVector(1, get(lapse));
  for (size_t i = 0; i < 3; ++i) {
    dv_shift.get(i) = DataVector(1, shift.get(i));
  }
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      dv_spatial_metric.get(i, j) = DataVector(1, spatial_metric.get(i, j));
    }
  }
  for (size_t i = 0; i < 3; ++i) {
    dv_unit_normal.get(i) = DataVector(1, unit_normal.get(i));
  }
  const auto char_speeds = characteristic_speeds(
      dv_rest_mass_density, dv_specific_internal_energy, dv_specific_enthalpy,
      dv_spatial_velocity, dv_lorentz_factor, dv_magnetic_field, dv_lapse,
      dv_shift, dv_spatial_metric, dv_unit_normal, equation_of_state);
  // This is the end of the hacky code block

  DataVector eigenvalues(9);
  for (size_t i = 0; i < 9; ++i) {
    eigenvalues[i] = gsl::at(char_speeds, i)[0];
  }

  Matrix right(9, 9);

  // We'd like to use `blaze::eigen` to get the eigenvalues and eigenvectors
  // of the flux Jacobian matrix `a`... but because `a` is not symmetric,
  // blaze generically produces complex eigenvectors. So instead we find the
  // nullspace of `a - \lambda I` using `blaze::svd`.
  blaze::DynamicMatrix<double, blaze::rowMajor> a_minus_lambda;
  blaze::DynamicMatrix<double, blaze::rowMajor> U;      // left singular vectors
  blaze::DynamicVector<double, blaze::columnVector> s;  // singular values
  blaze::DynamicMatrix<double, blaze::rowMajor> V;  // right singular vectors

  const auto find_group_of_eigenvectors =
      [&a_minus_lambda, &a, &U, &s, &V, &eigenvalues, &right](
          const size_t index, const size_t degeneracy) noexcept {
        a_minus_lambda = a;
        for (size_t i = 0; i < 9; ++i) {
          a_minus_lambda(i, i) -= eigenvalues[index];
        }
        blaze::svd(a_minus_lambda, U, s, V);

        // Check the null space has the expected size: the last degeneracy
        // singular values should vanish
#ifdef SPECTRE_DEBUG
        for (size_t i = 0; i < 9 - degeneracy; ++i) {
          ASSERT(fabs(s[i]) > 1e-14, "Bad SVD");
        }
        for (size_t i = 9 - degeneracy; i < 9; ++i) {
          ASSERT(fabs(s[i]) < 1e-14, "Bad SVD");
        }
#endif  // ifdef SPECTRE_DEBUG

        // Copy the last degeneracy rows of V into the
        // (index, index+degeneracy) columns of right
        for (size_t i = 0; i < 9; ++i) {
          for (size_t j = 0; j < degeneracy; ++j) {
            right(i, index + j) = V(9 - degeneracy + j, i);
          }
        }
      };

  // lambda = inward div-clean mode
  find_group_of_eigenvectors(0, 1);
  // lambda = inward fast mode
  find_group_of_eigenvectors(1, 1);
  // 5 degenerate eigenvalues, lambda = alfven/slow modes
  find_group_of_eigenvectors(2, 5);
  // lambda = outward fast mode
  find_group_of_eigenvectors(7, 1);
  // lambda = outward div-clean mode
  find_group_of_eigenvectors(8, 1);

  Matrix left = right;
  blaze::invert<blaze::asGeneral>(left);

  return std::make_pair(eigenvalues, std::make_pair(right, left));
}

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                              \
  template std::array<DataVector, 9> characteristic_speeds<GET_DIM(data)>(  \
      const Scalar<DataVector>& rest_mass_density,                          \
      const Scalar<DataVector>& specific_internal_energy,                   \
      const Scalar<DataVector>& specific_enthalpy,                          \
      const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,      \
      const Scalar<DataVector>& lorentz_factor,                             \
      const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,        \
      const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift, \
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,       \
      const tnsr::i<DataVector, 3>& unit_normal,                            \
      const EquationsOfState::EquationOfState<true, GET_DIM(data)>&         \
          equation_of_state) noexcept;                                      \
  template void characteristic_speeds<GET_DIM(data)>(                       \
      const gsl::not_null<std::array<DataVector, 9>*> char_speeds,          \
      const Scalar<DataVector>& rest_mass_density,                          \
      const Scalar<DataVector>& specific_internal_energy,                   \
      const Scalar<DataVector>& specific_enthalpy,                          \
      const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,      \
      const Scalar<DataVector>& lorentz_factor,                             \
      const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,        \
      const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift, \
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,       \
      const tnsr::i<DataVector, 3>& unit_normal,                            \
      const EquationsOfState::EquationOfState<true, GET_DIM(data)>&         \
          equation_of_state) noexcept;                                      \
  template std::pair<DataVector, std::pair<Matrix, Matrix>>                 \
  numerical_eigensystem<GET_DIM(data)>(                                     \
      const Scalar<double>& rest_mass_density,                              \
      const Scalar<double>& specific_internal_energy,                       \
      const Scalar<double>& specific_enthalpy,                              \
      const tnsr::I<double, 3, Frame::Inertial>& spatial_velocity,          \
      const Scalar<double>& lorentz_factor,                                 \
      const tnsr::I<double, 3, Frame::Inertial>& magnetic_field,            \
      const Scalar<double>& lapse, const tnsr::I<double, 3>& shift,         \
      const tnsr::ii<double, 3, Frame::Inertial>& spatial_metric,           \
      const tnsr::i<double, 3>& unit_normal,                                \
      const EquationsOfState::EquationOfState<true, GET_DIM(data)>&         \
          equation_of_state) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2))

#undef GET_DIM
#undef INSTANTIATION
}  // namespace grmhd::ValenciaDivClean
