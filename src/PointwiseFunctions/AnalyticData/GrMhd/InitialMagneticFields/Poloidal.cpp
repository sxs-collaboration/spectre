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
}

// NOLINTNEXTLINE
PUP::able::PUP_ID Poloidal::my_PUP_ID = 0;

Poloidal::Poloidal(const size_t pressure_exponent, const double cutoff_pressure,
                   const double vector_potential_amplitude)
    : pressure_exponent_(pressure_exponent),
      cutoff_pressure_(cutoff_pressure),
      vector_potential_amplitude_(vector_potential_amplitude) {}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::MagneticField<DataType, 3>>
Poloidal::variables(const tnsr::I<DataType, 3>& coords,
                    const Scalar<DataType>& pressure,
                    const Scalar<DataType>& sqrt_det_spatial_metric,
                    const tnsr::i<DataType, 3>& dcoords_pressure) const {
  auto magnetic_field = make_with_value<tnsr::I<DataType, 3>>(coords, 0.0);
  const size_t num_pts = get_size(get(pressure));

  for (size_t i = 0; i < num_pts; ++i) {
    const double pressure_i = get_element(get(pressure), i);
    if (pressure_i < cutoff_pressure_) {
      get_element(magnetic_field.get(0), i) = 0.0;
      get_element(magnetic_field.get(1), i) = 0.0;
      get_element(magnetic_field.get(2), i) = 0.0;
      continue;
    }

    // (p - p_c)^{n_s}
    const double pressure_term =
        pow(pressure_i - cutoff_pressure_, pressure_exponent_);
    // n_s * (p - p_c)^{n_s-1}
    const double n_times_pressure_to_n_minus_1 =
        pressure_exponent_ * pow(pressure_i - cutoff_pressure_,
                                 static_cast<int>(pressure_exponent_) - 1);

    const double x = get_element(coords.get(0), i);
    const double y = get_element(coords.get(1), i);

    const auto& dp_dx = get_element(dcoords_pressure.get(0), i);
    const auto& dp_dy = get_element(dcoords_pressure.get(1), i);
    const auto& dp_dz = get_element(dcoords_pressure.get(2), i);

    // Assign Bx, By, Bz
    get_element(magnetic_field.get(0), i) =
        -n_times_pressure_to_n_minus_1 * x * dp_dz;
    get_element(magnetic_field.get(1), i) =
        -n_times_pressure_to_n_minus_1 * y * dp_dz;
    get_element(magnetic_field.get(2), i) =
        2.0 * pressure_term +
        n_times_pressure_to_n_minus_1 * (x * dp_dx + y * dp_dy);
  }

  for (size_t d = 0; d < 3; ++d) {
    magnetic_field.get(d) *=
        vector_potential_amplitude_ / get(sqrt_det_spatial_metric);
  }

  return {std::move(magnetic_field)};
}

bool operator==(const Poloidal& lhs, const Poloidal& rhs) {
  return lhs.pressure_exponent_ == rhs.pressure_exponent_ and
         lhs.cutoff_pressure_ == rhs.cutoff_pressure_ and
         lhs.vector_potential_amplitude_ == rhs.vector_potential_amplitude_;
}

bool operator!=(const Poloidal& lhs, const Poloidal& rhs) {
  return not(lhs == rhs);
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                               \
  template tuples::TaggedTuple<hydro::Tags::MagneticField<DTYPE(data), 3>> \
  Poloidal::variables<DTYPE(data)>(                                        \
      const tnsr::I<DTYPE(data), 3>& coords,                               \
      const Scalar<DTYPE(data)>& pressure,                                 \
      const Scalar<DTYPE(data)>& sqrt_det_spatial_metric,                  \
      const tnsr::i<DTYPE(data), 3>& dcoords_pressure) const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef INSTANTIATE
#undef DTYPE

}  // namespace grmhd::AnalyticData::InitialMagneticFields
