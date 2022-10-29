// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/BoundaryConditions/DemandOutgoingCharSpeeds.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <pup.h>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/NewtonianEuler/SoundSpeedSquared.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/TMPL.hpp"

namespace NewtonianEuler::BoundaryConditions {
template <size_t Dim>
DemandOutgoingCharSpeeds<Dim>::DemandOutgoingCharSpeeds(
    CkMigrateMessage* const msg)
    : BoundaryCondition<Dim>(msg) {}

template <size_t Dim>
std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
DemandOutgoingCharSpeeds<Dim>::get_clone() const {
  return std::make_unique<DemandOutgoingCharSpeeds>(*this);
}

template <size_t Dim>
void DemandOutgoingCharSpeeds<Dim>::pup(PUP::er& p) {
  BoundaryCondition<Dim>::pup(p);
}

template <size_t Dim>
// NOLINTNEXTLINE
PUP::able::PUP_ID DemandOutgoingCharSpeeds<Dim>::my_PUP_ID = 0;

template <size_t Dim>
template <size_t ThermodynamicDim>
std::optional<std::string>
DemandOutgoingCharSpeeds<Dim>::dg_demand_outgoing_char_speeds(
    const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
        face_mesh_velocity,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        outward_directed_normal_covector,

    const Scalar<DataVector>& mass_density,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& velocity,
    const Scalar<DataVector>& specific_internal_energy,
    const EquationsOfState::EquationOfState<false, ThermodynamicDim>&
        equation_of_state) {
  double min_char_speed = std::numeric_limits<double>::signaling_NaN();

  // Temp buffer for sound speed, normal dot velocity, and (optional) normal
  // dot mesh velocity
  Variables<tmpl::list<::Tags::TempScalar<0>, ::Tags::TempScalar<1>,
                       ::Tags::TempScalar<2>>>
      buffer{get(mass_density).size()};

  auto& sound_speed = get<::Tags::TempScalar<0>>(buffer);
  sound_speed_squared(make_not_null(&sound_speed), mass_density,
                      specific_internal_energy, equation_of_state);
  get(sound_speed) = sqrt(get(sound_speed));

  auto& normal_dot_velocity = get<::Tags::TempScalar<1>>(buffer);
  dot_product(make_not_null(&normal_dot_velocity),
              outward_directed_normal_covector, velocity);

  if (face_mesh_velocity.has_value()) {
    auto& normal_dot_mesh_velocity = get<::Tags::TempScalar<2>>(buffer);
    dot_product(make_not_null(&normal_dot_mesh_velocity),
                outward_directed_normal_covector, face_mesh_velocity.value());

    min_char_speed = min(get(normal_dot_velocity) - get(sound_speed) -
                         get(normal_dot_mesh_velocity));
  } else {
    min_char_speed = min(get(normal_dot_velocity) - get(sound_speed));
  }

  if (min_char_speed < 0.0) {
    return {MakeString{}
            << "DemandOutgoingCharSpeeds boundary condition violated with the "
               "characteristic speed : "
            << min_char_speed << "\n"};
  }

  return std::nullopt;
}  // namespace NewtonianEuler::BoundaryConditions

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(_, data) \
  template class DemandOutgoingCharSpeeds<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION

#define THERMODIM(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATION(_, data)                                              \
  template std::optional<std::string>                                       \
  DemandOutgoingCharSpeeds<DIM(data)>::dg_demand_outgoing_char_speeds<      \
      THERMODIM(data)>(                                                     \
      const std::optional<tnsr::I<DataVector, DIM(data), Frame::Inertial>>& \
          face_mesh_velocity,                                               \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                \
          outward_directed_normal_covector,                                 \
      const Scalar<DataVector>& mass_density,                               \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& velocity,      \
      const Scalar<DataVector>& specific_internal_energy,                   \
      const EquationsOfState::EquationOfState<false, THERMODIM(data)>&      \
          equation_of_state);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3), (1, 2))

#undef INSTANTIATION

#undef THERMODIM

#undef DIM
}  // namespace NewtonianEuler::BoundaryConditions
