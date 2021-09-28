// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/CurvedScalarWave/BoundaryConditions/ConstraintPreservingSphericalRadiation.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/CurvedScalarWave/Characteristics.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace CurvedScalarWave::BoundaryConditions {

template <size_t Dim>
std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
ConstraintPreservingSphericalRadiation<Dim>::get_clone() const {
  return std::make_unique<ConstraintPreservingSphericalRadiation>(*this);
}

template <size_t Dim>
void ConstraintPreservingSphericalRadiation<Dim>::pup(PUP::er& p) {
  BoundaryCondition<Dim>::pup(p);
}

template <size_t Dim>
ConstraintPreservingSphericalRadiation<
    Dim>::ConstraintPreservingSphericalRadiation(CkMigrateMessage* const msg)
    : BoundaryCondition<Dim>(msg) {}

template <size_t Dim>
std::optional<std::string>
ConstraintPreservingSphericalRadiation<Dim>::dg_time_derivative(
    const gsl::not_null<Scalar<DataVector>*> dt_pi_correction,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
        dt_phi_correction,
    const gsl::not_null<Scalar<DataVector>*> dt_psi_correction,
    const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
        face_mesh_velocity,
    const tnsr::i<DataVector, Dim>& normal_covector,
    const tnsr::I<DataVector, Dim>& normal_vector,
    const tnsr::i<DataVector, Dim>& phi, const Scalar<DataVector>& psi,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& coords,
    const Scalar<DataVector>& gamma1, const Scalar<DataVector>& gamma2,
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, Dim>& shift,
    const Scalar<DataVector>& dt_pi, const tnsr::i<DataVector, Dim>& dt_phi,
    const Scalar<DataVector>& dt_psi, const tnsr::i<DataVector, Dim>& d_psi,
    const tnsr::ij<DataVector, Dim>& d_phi) const {
  Variables<tmpl::list<::Tags::Tempa<0, 3>, ::Tags::TempScalar<1>,
                       ::Tags::TempScalar<2>>>
      buffer{get<0>(coords).size()};

  auto& char_speeds = get<::Tags::Tempa<0, 3>>(buffer);
  auto& inv_radius = get<::Tags::TempScalar<1>>(buffer);
  auto& face_speed = get<::Tags::TempScalar<2>>(buffer);

  get(inv_radius) = 1. / get(magnitude(coords));
  characteristic_speeds(make_not_null(&char_speeds), gamma1, lapse, shift,
                        normal_covector);

  if (face_mesh_velocity.has_value()) {
    face_speed = dot_product(normal_covector, *face_mesh_velocity);
    for (size_t i = 0; i < 4; ++i) {
      char_speeds.get(i) -= get(face_speed);
    }
  }

  get(*dt_psi_correction) =
      get<0>(normal_vector) * (get<0>(d_psi) - get<0>(phi));
  for (size_t i = 1; i < Dim; ++i) {
    get(*dt_psi_correction) +=
        normal_vector.get(i) * (d_psi.get(i) - phi.get(i));
  }

  for (size_t i = 0; i < Dim; ++i) {
    dt_phi_correction->get(i) =
        get<0>(normal_vector) * (d_phi.get(0, i) - d_phi.get(i, 0));
    for (size_t j = 1; j < Dim; ++j) {
      dt_phi_correction->get(i) +=
          normal_vector.get(j) * (d_phi.get(j, i) - d_phi.get(i, j));
    }
  }

  for (size_t i = 0; i < get(*dt_psi_correction).size(); ++i) {
    get(*dt_psi_correction)[i] *= std::min(0., gsl::at(char_speeds[0], i));
    for (size_t j = 0; j < Dim; ++j) {
      dt_phi_correction->get(j)[i] *=
          0.5 * std::min(0., gsl::at(char_speeds[1], i));
    }
  }

  get(*dt_pi_correction) =
      (2.0 * get(inv_radius) * get(psi) + 4.0 * get(dt_psi)) * get(inv_radius);
  for (size_t i = 0; i < Dim; ++i) {
    get(*dt_pi_correction) +=
        normal_vector.get(i) *
            (2. * dt_phi.get(i) + 4.0 * get(inv_radius) * phi.get(i)) +
        shift.get(i) * dt_phi.get(i);
    for (size_t j = 0; j < Dim; ++j) {
      get(*dt_pi_correction) +=
          normal_vector.get(i) * normal_vector.get(j) * d_phi.get(i, j);
    }
  }
  get(*dt_pi_correction) /= get(lapse);
  get(*dt_pi_correction) += get(gamma2) * get(*dt_psi_correction) - get(dt_pi);
  return {};
}

template <size_t Dim>
// NOLINTNEXTLINE
PUP::able::PUP_ID ConstraintPreservingSphericalRadiation<Dim>::my_PUP_ID = 0;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data) \
  template class ConstraintPreservingSphericalRadiation<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace CurvedScalarWave::BoundaryConditions
