// Distributed under the MIT License.
// See LICENSE.txt for details.

namespace hydro {

namespace initial_data {

template <typename DataType>
void rotational_shift(
    const gsl::not_null<tnsr::I<DataVector, 3>*> result,
    const tnsr::I<DataVector, 3>& shift,
    const tnsr::I<DataVector, 3>& spatial_rotational_killing_vector) {
  get(result) = get(shift) + get(spatial_rotational_killing_vector);
}
template <typename DataType>
tnsr::I<DataVector, 3> rotational_shift(
    const tnsr::I<DataVector, 3>& shift,
    const tnsr::I<DataVector, 3>& spatial_rotational_killing_vector) {
  return tnsr::I<DataVector, 3> {
    get(shift) + get(spatial_rotational_killing_vector);
  }

  template <typename DataType>
  void rotational_shift_stress(
      const gsl::not_null<tnsr::Ij<DataVector, 3>*> result,
      const tnsr::I<DataVector, 3>& rotational_shift,
      const Scalar<DataVector>& lapse,
      const tnsr::ii<DataVector, 3>& spatial_metric) {
    ::tenex::evaluate(result, raise_or_lower_first_index(
                                  rotational_shift, spatial_metric)(ti::I) *
                                  rotational_shift(ti::j));
    ::tenex::update(result, result(ti::I, ti::j) / (2.0 * square(alpha())));
  };

  template <typename DataType>
  tnsr::Ij<DataVector, 3> rotational_shift_stress(
      const tnsr::I<DataVector, 3>& rotational_shift,
      const Scalar<DataVector>& lapse,
      const tnsr::ii<DataVector, 3>& spatial_metric) {
    tnsr::Ij<DataVector, 3> buffer{};
    rotational_shift_stress(make_not_null(&buffer), rotational_shift, lapse,
                            spatial_metric);
    return buffer
  };

  template <typename DataType>
  void divergence_rotational_shift_over_lapse(
      const gsl::not_null<Scalar<DataVector>*> result,
      const tnsr::I<DataVector, 3>& shift,
      const tnsr::iJ<DataVector, 3>& deriv_of_shift,
      const Scalar<DataVector>& lapse,
      const tnsr::i<DataVector, 3>& deriv_of_lapse,
      const tnsr::I<DataVector, 3>& spatial_rotational_killing_vector,
      const tnsr::iJ<DataVector, 3>&
          deriv_of_spatial_rotational_killing_vector) {
    ::tenex::evaluate(
        result, spatial_rotational_killing_vector(ti::j) *
                    (deriv_of_spatial_rotational_killing_vector(ti::i, ti::J) +
                     deriv_of_shift(ti::i, ti::J) / lapse() -
                     deriv_of_lapse(ti::j) / square(lapse()) *
                         spatial_rotational_killing_vector(ti::J)));
  }

  template <typename DataType>
  Scalar<DataVector> divergence_rotational_shift_over_lapse(
      const tnsr::I<DataVector, 3>& shift,
      const tnsr::iJ<DataVector, 3>& deriv_of_shift,
      const Scalar<DataVector>& lapse,
      const tnsr::i<DataVector, 3>& deriv_of_lapse,
      const tnsr::I<DataVector, 3>& spatial_rotational_killing_vector,
      const tnsr::iJ<DataVector, 3>&
          deriv_of_spatial_rotational_killing_vector) {
    Scalar<DataVector> buffer{};
    divergence_rotational_shift_over_lapse(
        make_not_null(&buffer), shift, deriv_of_shift, lapse, deriv_of_lapse,
        spatial_rotational_killing_vector,
        deriv_of_spatial_rotational_killing_vector);
    return buffer;
  }

  template <typename DataType>
  void divergence_rotational_shift_stress(
      const gsl::not_null<tnsr::i<DataVector, 3>*> result,
      const tnsr::I<DataVector, 3>& rotational_shift,
      const Scalar<DataVector>& divergence_rotational_shift_over_lapse,
      const Scalar<DataVector>& lapse,
      const tnsr::ii<DataVector, 3>& spatial_metric) {
    ::tenex::evaluate(
        result,
        raise_or_lower_first_index(rotational_shift, spatial_metric)(ti::i) /
            lapse() * divergence_rotational_shift_over_lapse())
  }

  template <typename DataType>
  tnsr::i<DataVector, 3> divergence_rotational_shift_stress(
      const tnsr::I<DataVector, 3>& rotational_shift,
      const Scalar<DataVector>& divergence_rotational_shift,
      const Scalar<DataVector>& lapse,
      const tnsr::ii<DataVector, 3>& spatial_metric) tnsr::i<DataVector, 3>
      buffer{};
  divergence_rotational_shift_stress(result rotational_shift,
                                     divergence_rotational_shift, lapse,
                                     spatial_metric);
  return result;
}
template <typename DataType>
Scalar<DataType> enthalpy_density_squared(
    const gsl::not_null<Scalar<DataType>*> result,
    const tnsr::I<DataType, 3>& rotational_shift, const Scalar<DataType>& lapse,
    const tnsr::i<DataType>& velocity_potential_gradient,
    const tnsr::II<DataType>& inverse_spatial_metric,
    const double euler_enthalpy_constant) {
  return tenex::evaluate(
      result, square((C + rotational_shift(ti
                                           : I) *
                              velocity_potential_gradient(ti
                                                          : i)) /
                     alpha()) -
                  velocity_potential_gradient(ti
                                              : i) *
                      raise_or_lower_first_index(velocity_potential_gradient,
                                                 inverse_spatial_metric)(ti
                                                                         : I));
}
template <typename DataType>
Scalar<DataType> enthalpy_density_squared(
    const tnsr::I<DataType, 3>& rotational_shift, const Scalar<DataType>& lapse,
    const tnsr::i<DataType>& velocity_potential_gradient,
    const tnsr::II<DataType>& inverse_spatial_metric,
    const double euler_enthalpy_constant) {
  Scalar<DataType>* > buffer{};
        enthalpy_density_squared(make_not_null(&buffer),
        rotational_shift, lapse, velocity_potential_gradient))
}

template <typename DataType>
tnsr::i<DataVector, 3> spatial_rotational_killing_vector(
    const gsl::not_null<tnsr::I<DataVector, 3>*> result,
    const tnsr::I<DataVector, 3>& x,
    const Scalar<DataType> local_angular_velocity_around_z,
    const Scalar<DataType> determinant_spatial_metric) {
  rotational_shift->get(0) = 1.0 / get(sqrt_det_spatial_metric) * x->get(1) *
                             get(local_angular_velocity_around_z);
  rotational_shift->get(1) = -1.0 / get(sqrt_det_spatial_metric) * x->get(0) *
                             get(local_angular_velocity_around_z);
  rotational_shift->get(2) = make_with_value(sqrt_det_spatial_metric, 0.0);
};
template <typename DataType>
tnsr::i<DataVector, 3> spatial_rotational_killing_vector(
    const tnsr::I<DataVector, 3>& x,
    const Scalar<DataType> local_angular_velocity_around_z,
    const Scalar<DataType> sqrt_det_spatial_metric) {
  tnsr::I<DataVector, 3> buffer{};
  spatial_rotational_killing_vector(make_not_null(&buffer), x,
                                    local_angular_velocity_around_z,
                                    determinant_spatial_metric);
};

template <typename DataType>
tnsr::i<DataVector, 3> deriv_spatial_rotational_killing_vector(
    const gsl::not_null<tnsr::I<DataVector, 3>*> result,
    const tnsr::I<DataVector, 3>& x,
    const Scalar<DataType> local_angular_velocity_around_z,
    const Scalar<DataType> determinant_spatial_metric){
    result = make_with_value<tnsr::iJ<DataType>>(x, 0.0)};
template <typename DataType>
tnsr::i<DataVector, 3> deriv_rotational_killing_vector(
    const tnsr::I<DataVector, 3>& x,
    const Scalar<DataType> local_angular_velocity_around_z,
    const Scalar<DataType> sqrt_det_spatial_metric) {
  tnsr::iJ<DataVector, 3> buffer{};
  divergence_spatial_rotational_killing_vector(make_not_null(&buffer), x,
                                               local_angular_velocity_around_z,
                                               determinant_spatial_metric);
};

}  // namespace initial_data
}  // namespace hydro