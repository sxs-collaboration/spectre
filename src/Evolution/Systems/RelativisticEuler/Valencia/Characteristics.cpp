// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/RelativisticEuler/Valencia/Characteristics.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/EagerMath/OrthonormalOneform.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Overloader.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare Tensor

/// \cond
namespace RelativisticEuler {
namespace Valencia {

template <size_t Dim>
void characteristic_speeds(
    const gsl::not_null<std::array<DataVector, Dim + 2>*> char_speeds,
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, Dim>& shift,
    const tnsr::I<DataVector, Dim>& spatial_velocity,
    const Scalar<DataVector>& spatial_velocity_squared,
    const Scalar<DataVector>& sound_speed_squared,
    const tnsr::i<DataVector, Dim>& normal) noexcept {
  const size_t num_grid_points = get<0>(shift).size();
  // Allocating a single large buffer is much faster than many small buffers
  Variables<tmpl::list<::Tags::TempScalar<0>, ::Tags::TempScalar<1>,
                       ::Tags::TempScalar<2>, ::Tags::TempScalar<3>,
                       ::Tags::TempScalar<4>>>
      temp_tensors{num_grid_points};

  // Because we don't require char_speeds to be of the correct size we use a
  // temp buffer for the dot product, then multiply by -1 assigning the result
  // to char_speeds.
  {
    Scalar<DataVector>& normal_shift = get<::Tags::TempScalar<0>>(temp_tensors);
    dot_product(make_not_null(&normal_shift), normal, shift);
    (*char_speeds)[0] = -1.0 * get(normal_shift);
  }
  // Dim-fold degenerate eigenvalue, reuse normal_shift allocation
  Scalar<DataVector>& normal_velocity =
      get<::Tags::TempScalar<0>>(temp_tensors);
  dot_product(make_not_null(&normal_velocity), normal, spatial_velocity);
  (*char_speeds)[1] = (*char_speeds)[0] + get(lapse) * get(normal_velocity);
  for (size_t i = 2; i < Dim + 1; ++i) {
    gsl::at(*char_speeds, i) = (*char_speeds)[1];
  }

  Scalar<DataVector>& one_minus_v_sqrd_cs_sqrd =
      get<::Tags::TempScalar<1>>(temp_tensors);
  get(one_minus_v_sqrd_cs_sqrd) =
      1.0 - get(spatial_velocity_squared) * get(sound_speed_squared);
  Scalar<DataVector>& vn_times_one_minus_cs_sqrd =
      get<::Tags::TempScalar<2>>(temp_tensors);
  get(vn_times_one_minus_cs_sqrd) =
      get(normal_velocity) * (1.0 - get(sound_speed_squared));

  Scalar<DataVector>& first_term = get<::Tags::TempScalar<3>>(temp_tensors);
  get(first_term) = get(lapse) / get(one_minus_v_sqrd_cs_sqrd);
  Scalar<DataVector>& second_term = get<::Tags::TempScalar<4>>(temp_tensors);
  get(second_term) =
      get(first_term) * sqrt(get(sound_speed_squared)) *
      sqrt((1.0 - get(spatial_velocity_squared)) *
           (get(one_minus_v_sqrd_cs_sqrd) -
            get(normal_velocity) * get(vn_times_one_minus_cs_sqrd)));
  get(first_term) *= get(vn_times_one_minus_cs_sqrd);

  (*char_speeds)[Dim + 1] =
      (*char_speeds)[0] + get(first_term) + get(second_term);
  (*char_speeds)[0] += get(first_term) - get(second_term);
}

template <size_t Dim>
std::array<DataVector, Dim + 2> characteristic_speeds(
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, Dim>& shift,
    const tnsr::I<DataVector, Dim>& spatial_velocity,
    const Scalar<DataVector>& spatial_velocity_squared,
    const Scalar<DataVector>& sound_speed_squared,
    const tnsr::i<DataVector, Dim>& normal) noexcept {
  std::array<DataVector, Dim + 2> char_speeds{};
  characteristic_speeds(make_not_null(&char_speeds), lapse, shift,
                        spatial_velocity, spatial_velocity_squared,
                        sound_speed_squared, normal);
  return char_speeds;
}

template <size_t Dim>
Matrix right_eigenvectors(const Scalar<double>& rest_mass_density,
                          const tnsr::I<double, Dim>& spatial_velocity,
                          const Scalar<double>& specific_internal_energy,
                          const Scalar<double>& pressure,
                          const Scalar<double>& specific_enthalpy,
                          const Scalar<double>& kappa_over_density,
                          const Scalar<double>& sound_speed_squared,
                          const Scalar<double>& lorentz_factor,
                          const tnsr::ii<double, Dim>& spatial_metric,
                          const tnsr::II<double, Dim>& inv_spatial_metric,
                          const Scalar<double>& det_spatial_metric,
                          const tnsr::i<double, Dim>& unit_normal) noexcept {
  const auto spatial_velocity_oneform =
      raise_or_lower_index(spatial_velocity, spatial_metric);

  const double normal_velocity =
      get(dot_product(unit_normal, spatial_velocity));
  const double velocity_squared =
      get(dot_product(spatial_velocity, spatial_velocity_oneform));

  // Use expression for (hW - 1) with well-behaved Newtonian limit
  const double h_w_minus_one =
      get(lorentz_factor) *
      (get(specific_internal_energy) + get(pressure) / get(rest_mass_density) +
       get(lorentz_factor) * velocity_squared / (get(lorentz_factor) + 1.0));

  const double factor_d =
      get(lorentz_factor) *
      sqrt(1.0 - velocity_squared * get(sound_speed_squared) -
           square(normal_velocity) * (1.0 - get(sound_speed_squared)));
  const double sound_speed_over_d = sqrt(get(sound_speed_squared)) / factor_d;

  const double specific_enthalpy_times_lorentz_factor =
      get(specific_enthalpy) * get(lorentz_factor);

  const double h_w_vn_c_over_d = specific_enthalpy_times_lorentz_factor *
                                 normal_velocity * sound_speed_over_d;

  Matrix result(Dim + 2, Dim + 2);

  // set eigenvector with eigenvalue = lapse * Lambda_minus - normal_shift
  size_t eigenvector_id = 0;

  result(0, eigenvector_id) = 1.0;
  result(1, eigenvector_id) = h_w_minus_one - h_w_vn_c_over_d;
  for (size_t i = 0; i < Dim; ++i) {
    result(i + 2, eigenvector_id) = specific_enthalpy_times_lorentz_factor *
                                    (spatial_velocity_oneform.get(i) -
                                     sound_speed_over_d * unit_normal.get(i));
  }

  // set eigenvector with eigenvalue = lapse * Lambda_plus - normal_shift
  eigenvector_id = Dim + 1;

  result(0, eigenvector_id) = 1.0;
  result(1, eigenvector_id) = h_w_minus_one + h_w_vn_c_over_d;
  for (size_t i = 0; i < Dim; ++i) {
    result(i + 2, eigenvector_id) = specific_enthalpy_times_lorentz_factor *
                                    (spatial_velocity_oneform.get(i) +
                                     sound_speed_over_d * unit_normal.get(i));
  }

  // set eigenvector with eigenvalue = lapse * normal_velocity - normal_shift
  eigenvector_id = Dim;

  result(0, eigenvector_id) = get(kappa_over_density);
  result(1, eigenvector_id) =
      get(kappa_over_density) * h_w_minus_one -
      specific_enthalpy_times_lorentz_factor * get(sound_speed_squared);

  const double prefactor = specific_enthalpy_times_lorentz_factor *
                           (get(kappa_over_density) - get(sound_speed_squared));
  for (size_t i = 0; i < Dim; ++i) {
    result(i + 2, eigenvector_id) = prefactor * spatial_velocity_oneform.get(i);
  }

  // if Dim > 1 set degenerate eigenvectors
  if (Dim > 1) {
    const double two_h_w_minus_one =
        h_w_minus_one + specific_enthalpy_times_lorentz_factor;
    const auto unit_tangent_oneform =
        orthonormal_oneform(unit_normal, inv_spatial_metric);
    const auto set_degenerate_eigenvector =
        [
          &result, &lorentz_factor, &spatial_velocity,
          &spatial_velocity_oneform, &two_h_w_minus_one, &specific_enthalpy
        ](const size_t vector_id,
          const tnsr::i<double, Dim>& tangent_oneform) noexcept {
      result(0, vector_id) =
          get(lorentz_factor) *
          get(dot_product(tangent_oneform, spatial_velocity));
      result(1, vector_id) = two_h_w_minus_one * result(0, vector_id);
      const double local_prefactor =
          2.0 * get(lorentz_factor) * result(0, vector_id);
      for (size_t i = 0; i < Dim; ++i) {
        result(i + 2, vector_id) =
            get(specific_enthalpy) *
            (tangent_oneform.get(i) +
             local_prefactor * spatial_velocity_oneform.get(i));
      }
    };
    set_degenerate_eigenvector(1, unit_tangent_oneform);

    if (Dim > 2) {
      // orthonormal_oneform for two forms is defined for 3-d only
      // so it needs to be called from a make_overloader.
      make_overloader([](std::integral_constant<size_t, 1> /*dim*/,
                         const auto&...) noexcept {},
                      [](std::integral_constant<size_t, 2> /*dim*/,
                         const auto&...) noexcept {},
                      [&set_degenerate_eigenvector](
                          std::integral_constant<size_t, 3> /*dim*/,
                          const auto& the_unit_normal,
                          const auto& the_unit_tangent_oneform,
                          const auto& the_spatial_metric,
                          const auto& the_det_spatial_metric) noexcept {
                        set_degenerate_eigenvector(
                            2, orthonormal_oneform<double, Frame::Inertial>(
                                   the_unit_normal, the_unit_tangent_oneform,
                                   the_spatial_metric, the_det_spatial_metric));
                      })(std::integral_constant<size_t, Dim>{}, unit_normal,
                         unit_tangent_oneform, spatial_metric,
                         det_spatial_metric);
    }
  }

  return result;
}

template <size_t Dim>
Matrix left_eigenvectors(const Scalar<double>& rest_mass_density,
                         const tnsr::I<double, Dim>& spatial_velocity,
                         const Scalar<double>& specific_internal_energy,
                         const Scalar<double>& pressure,
                         const Scalar<double>& specific_enthalpy,
                         const Scalar<double>& kappa_over_density,
                         const Scalar<double>& sound_speed_squared,
                         const Scalar<double>& lorentz_factor,
                         const tnsr::ii<double, Dim>& spatial_metric,
                         const tnsr::II<double, Dim>& inv_spatial_metric,
                         const Scalar<double>& det_spatial_metric,
                         const tnsr::i<double, Dim>& unit_normal) noexcept {
  const auto unit_normal_vector =
      raise_or_lower_index(unit_normal, inv_spatial_metric);

  const double inv_specific_enthalpy = 1.0 / get(specific_enthalpy);
  const double normal_velocity =
      get(dot_product(unit_normal, spatial_velocity));
  const double vn_squared = square(normal_velocity);
  const double v_squared =
      get(dot_product(spatial_velocity, spatial_velocity, spatial_metric));

  const double sound_speed_times_d =
      sqrt(get(sound_speed_squared)) * get(lorentz_factor) *
      sqrt(1.0 - get(sound_speed_squared) * v_squared -
           vn_squared * (1.0 - get(sound_speed_squared)));
  const double one_minus_vn_squared = 1.0 - vn_squared;

  // Use expression for (h - W) with well-behaved Newtonian limit
  const double h_minus_w =
      get(specific_internal_energy) + get(pressure) / get(rest_mass_density) -
      square(get(lorentz_factor)) * v_squared / (get(lorentz_factor) + 1.0);

  Matrix result(Dim + 2, Dim + 2);

  // set eigenvector with eigenvalue = lapse * Lambda_minus - normal_shift
  size_t eigenvector_id = 0;

  const double terms_with_h_and_w =
      get(lorentz_factor) * one_minus_vn_squared *
      (get(sound_speed_squared) *
           (get(specific_enthalpy) + get(lorentz_factor)) -
       get(kappa_over_density) * h_minus_w);

  const double common_prefactor = 0.5 * inv_specific_enthalpy /
                                  get(sound_speed_squared) /
                                  get(lorentz_factor) / one_minus_vn_squared;

  const double terms_with_k_and_c =
      common_prefactor * square(get(lorentz_factor)) * one_minus_vn_squared *
      (get(kappa_over_density) + get(sound_speed_squared));

  result(eigenvector_id, 0) =
      -common_prefactor *
      (get(sound_speed_squared) - normal_velocity * sound_speed_times_d);
  result(eigenvector_id, 1) = terms_with_k_and_c + result(eigenvector_id, 0);
  result(eigenvector_id, 0) += (common_prefactor * terms_with_h_and_w);

  // will reuse later
  double prefactor_for_normal_vector =
      common_prefactor *
      (get(sound_speed_squared) * normal_velocity - sound_speed_times_d);
  for (size_t i = 0; i < Dim; ++i) {
    result(eigenvector_id, i + 2) =
        prefactor_for_normal_vector * unit_normal_vector.get(i) -
        terms_with_k_and_c * spatial_velocity.get(i);
  }

  // set eigenvector with eigenvalue = lapse * Lambda_plus - normal_shift
  eigenvector_id = Dim + 1;

  result(eigenvector_id, 0) =
      -common_prefactor *
      (get(sound_speed_squared) + normal_velocity * sound_speed_times_d);
  result(eigenvector_id, 1) = terms_with_k_and_c + result(eigenvector_id, 0);
  result(eigenvector_id, 0) += (common_prefactor * terms_with_h_and_w);

  prefactor_for_normal_vector =
      common_prefactor *
      (get(sound_speed_squared) * normal_velocity + sound_speed_times_d);
  for (size_t i = 0; i < Dim; ++i) {
    result(eigenvector_id, i + 2) =
        prefactor_for_normal_vector * unit_normal_vector.get(i) -
        terms_with_k_and_c * spatial_velocity.get(i);
  }

  // set eigenvector with eigenvalue = lapse * normal_velocity - normal_shift
  eigenvector_id = Dim;

  const double inv_h_c_squared =
      inv_specific_enthalpy / get(sound_speed_squared);
  result(eigenvector_id, 0) = inv_h_c_squared * h_minus_w;
  result(eigenvector_id, 1) = inv_h_c_squared * get(lorentz_factor);
  for (size_t i = 0; i < Dim; ++i) {
    result(eigenvector_id, i + 2) =
        result(eigenvector_id, 1) * spatial_velocity.get(i);
  }
  result(eigenvector_id, 1) *= -1.0;

  // if Dim > 1 set degenerate eigenvectors
  if (Dim > 1) {
    const double prefactor = -inv_specific_enthalpy / one_minus_vn_squared;
    const auto unit_tangent_oneform =
        orthonormal_oneform(unit_normal, inv_spatial_metric);
    const auto set_degenerate_eigenvector =
        [
          &result, &prefactor, &spatial_velocity, &inv_spatial_metric,
          &normal_velocity, &inv_specific_enthalpy, &unit_normal_vector
        ](const size_t vector_id,
          const tnsr::i<double, Dim>& tangent_oneform) noexcept {
      result(vector_id, 0) =
          prefactor * get(dot_product(tangent_oneform, spatial_velocity));
      result(vector_id, 1) = result(vector_id, 0);
      const auto unit_tangent =
          raise_or_lower_index(tangent_oneform, inv_spatial_metric);
      const double local_prefactor = result(vector_id, 0) * normal_velocity;
      for (size_t i = 0; i < Dim; ++i) {
        result(vector_id, i + 2) = inv_specific_enthalpy * unit_tangent.get(i) -
                                   local_prefactor * unit_normal_vector.get(i);
      }
    };
    set_degenerate_eigenvector(1, unit_tangent_oneform);

    if (Dim > 2) {
      // orthonormal_oneform for two forms is defined for 3-d only
      // so it needs to be called from a make_overloader.
      make_overloader([](std::integral_constant<size_t, 1> /*dim*/,
                         const auto&...) noexcept {},
                      [](std::integral_constant<size_t, 2> /*dim*/,
                         const auto&...) noexcept {},
                      [&set_degenerate_eigenvector](
                          std::integral_constant<size_t, 3> /*dim*/,
                          const auto& the_unit_normal,
                          const auto& the_unit_tangent_oneform,
                          const auto& the_spatial_metric,
                          const auto& the_det_spatial_metric) noexcept {
                        set_degenerate_eigenvector(
                            2, orthonormal_oneform<double, Frame::Inertial>(
                                   the_unit_normal, the_unit_tangent_oneform,
                                   the_spatial_metric, the_det_spatial_metric));
                      })(std::integral_constant<size_t, Dim>{}, unit_normal,
                         unit_tangent_oneform, spatial_metric,
                         det_spatial_metric);
    }
  }

  return result;
}

}  // namespace Valencia
}  // namespace RelativisticEuler

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                   \
  template std::array<DataVector, DIM(data) + 2>                               \
  RelativisticEuler::Valencia::characteristic_speeds(                          \
      const Scalar<DataVector>& lapse,                                         \
      const tnsr::I<DataVector, DIM(data)>& shift,                             \
      const tnsr::I<DataVector, DIM(data)>& spatial_velocity,                  \
      const Scalar<DataVector>& spatial_velocity_squared,                      \
      const Scalar<DataVector>& sound_speed_squared,                           \
      const tnsr::i<DataVector, DIM(data)>& normal) noexcept;                  \
  template Matrix RelativisticEuler::Valencia::right_eigenvectors(             \
      const Scalar<double>& rest_mass_density,                                 \
      const tnsr::I<double, DIM(data)>& spatial_velocity,                      \
      const Scalar<double>& specific_internal_energy,                          \
      const Scalar<double>& pressure, const Scalar<double>& specific_enthalpy, \
      const Scalar<double>& kappa_over_density,                                \
      const Scalar<double>& sound_speed_squared,                               \
      const Scalar<double>& lorentz_factor,                                    \
      const tnsr::ii<double, DIM(data)>& spatial_metric,                       \
      const tnsr::II<double, DIM(data)>& inv_spatial_metric,                   \
      const Scalar<double>& det_spatial_metric,                                \
      const tnsr::i<double, DIM(data)>& unit_normal) noexcept;                 \
  template Matrix RelativisticEuler::Valencia::left_eigenvectors(              \
      const Scalar<double>& rest_mass_density,                                 \
      const tnsr::I<double, DIM(data)>& spatial_velocity,                      \
      const Scalar<double>& specific_internal_energy,                          \
      const Scalar<double>& pressure, const Scalar<double>& specific_enthalpy, \
      const Scalar<double>& kappa_over_density,                                \
      const Scalar<double>& sound_speed_squared,                               \
      const Scalar<double>& lorentz_factor,                                    \
      const tnsr::ii<double, DIM(data)>& spatial_metric,                       \
      const tnsr::II<double, DIM(data)>& inv_spatial_metric,                   \
      const Scalar<double>& det_spatial_metric,                                \
      const tnsr::i<double, DIM(data)>& unit_normal) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE
/// \endcond
