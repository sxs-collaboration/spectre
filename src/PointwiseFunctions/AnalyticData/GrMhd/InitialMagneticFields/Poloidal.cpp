// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/GrMhd/InitialMagneticFields/Poloidal.hpp"

#include <memory>

#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/InitialMagneticFields/InitialMagneticField.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace grmhd::AnalyticData::InitialMagneticFields {

std::unique_ptr<InitialMagneticField> Poloidal::get_clone() const {
  return std::make_unique<Poloidal>(*this);
}

Poloidal::Poloidal(CkMigrateMessage* msg) : InitialMagneticField(msg) {}

void Poloidal::pup(PUP::er& p) {
  InitialMagneticField::pup(p);
  p | pressure_exponent_;
  p | cutoff_pressure_;
  p | vector_potential_amplitude_;
  p | center_;
  p | max_distance_from_center_;
}

// NOLINTNEXTLINE
PUP::able::PUP_ID Poloidal::my_PUP_ID = 0;

Poloidal::Poloidal(const size_t pressure_exponent, const double cutoff_pressure,
                   const double vector_potential_amplitude,
                   const std::array<double, 3> center,
                   const double max_distance_from_center)
    : pressure_exponent_(pressure_exponent),
      cutoff_pressure_(cutoff_pressure),
      vector_potential_amplitude_(vector_potential_amplitude),
      center_(center),
      max_distance_from_center_(max_distance_from_center) {}

void Poloidal::variables(const gsl::not_null<tnsr::I<DataVector, 3>*> result,
                         const tnsr::I<DataVector, 3>& coords,
                         const Scalar<DataVector>& pressure,
                         const Scalar<DataVector>& sqrt_det_spatial_metric,
                         const tnsr::i<DataVector, 3>& deriv_pressure) const {
  ASSERT(result->get(0).size() == get(pressure).size(),
         "Result must be of size " << get(pressure).size() << " but got "
                                   << result->get(0).size());
  variables_impl(result, coords, pressure, sqrt_det_spatial_metric,
                 deriv_pressure);
}

void Poloidal::variables(const gsl::not_null<tnsr::I<double, 3>*> result,
                         const tnsr::I<double, 3>& coords,
                         const Scalar<double>& pressure,
                         const Scalar<double>& sqrt_det_spatial_metric,
                         const tnsr::i<double, 3>& deriv_pressure) const {
  variables_impl(result, coords, pressure, sqrt_det_spatial_metric,
                 deriv_pressure);
}

bool Poloidal::is_equal(const InitialMagneticField& rhs) const {
  const auto& derived_ptr = dynamic_cast<const Poloidal* const>(&rhs);
  return derived_ptr != nullptr and *derived_ptr == *this;
}

template <typename DataType>
void Poloidal::variables_impl(
    const gsl::not_null<tnsr::I<DataType, 3>*> magnetic_field,
    const tnsr::I<DataType, 3>& coords, const Scalar<DataType>& pressure,
    const Scalar<DataType>& sqrt_det_spatial_metric,
    const tnsr::i<DataType, 3>& deriv_pressure) const {
  const size_t num_pts = get_size(get(pressure));

  for (size_t i = 0; i < num_pts; ++i) {
    const double pressure_i = get_element(get(pressure), i);
    const double x = get_element(coords.get(0), i) - center_[0];
    const double y = get_element(coords.get(1), i) - center_[1];
    const double z = get_element(coords.get(2), i) - center_[2];
    const double radius = sqrt(x * x + y * y + z * z);
    if (pressure_i < cutoff_pressure_ or radius > max_distance_from_center_) {
      continue;
    }

    // (p - p_c)^{n_s}
    const double pressure_term = pow(pressure_i - cutoff_pressure_,
                                     static_cast<int>(pressure_exponent_));
    // n_s * (p - p_c)^{n_s-1}
    const double n_times_pressure_to_n_minus_1 =
        static_cast<double>(pressure_exponent_) *
        pow(pressure_i - cutoff_pressure_,
            static_cast<int>(pressure_exponent_) - 1);

    const auto& dp_dx = get_element(deriv_pressure.get(0), i);
    const auto& dp_dy = get_element(deriv_pressure.get(1), i);
    const auto& dp_dz = get_element(deriv_pressure.get(2), i);

    // Assign Bx, By, Bz
    get_element(magnetic_field->get(0), i) +=
        vector_potential_amplitude_ *
        (-n_times_pressure_to_n_minus_1 * x * dp_dz) /
        get_element(get(sqrt_det_spatial_metric), i);
    get_element(magnetic_field->get(1), i) +=
        vector_potential_amplitude_ *
        (-n_times_pressure_to_n_minus_1 * y * dp_dz) /
        get_element(get(sqrt_det_spatial_metric), i);
    get_element(magnetic_field->get(2), i) +=
        vector_potential_amplitude_ *
        (2.0 * pressure_term +
         n_times_pressure_to_n_minus_1 * (x * dp_dx + y * dp_dy)) /
        get_element(get(sqrt_det_spatial_metric), i);
  }
}

bool operator==(const Poloidal& lhs, const Poloidal& rhs) {
  return lhs.pressure_exponent_ == rhs.pressure_exponent_ and
         lhs.cutoff_pressure_ == rhs.cutoff_pressure_ and
         lhs.vector_potential_amplitude_ == rhs.vector_potential_amplitude_ and
         lhs.center_ == rhs.center_ and
         lhs.max_distance_from_center_ == rhs.max_distance_from_center_;
}

bool operator!=(const Poloidal& lhs, const Poloidal& rhs) {
  return not(lhs == rhs);
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                  \
  template void Poloidal::variables_impl<DTYPE(data)>(        \
      gsl::not_null<tnsr::I<DTYPE(data), 3>*> magnetic_field, \
      const tnsr::I<DTYPE(data), 3>& coords,                  \
      const Scalar<DTYPE(data)>& pressure,                    \
      const Scalar<DTYPE(data)>& sqrt_det_spatial_metric,     \
      const tnsr::i<DTYPE(data), 3>& deriv_pressure) const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef INSTANTIATE
#undef DTYPE

}  // namespace grmhd::AnalyticData::InitialMagneticFields
