// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/CurvedScalarWave/BoundaryConditions/Outflow.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/CurvedScalarWave/Characteristics.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeString.hpp"

namespace CurvedScalarWave::BoundaryConditions {

template <size_t Dim>
std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
Outflow<Dim>::get_clone() const noexcept {
  return std::make_unique<Outflow>(*this);
}

template <size_t Dim>
void Outflow<Dim>::pup(PUP::er& p) {
  BoundaryCondition<Dim>::pup(p);
}

template <size_t Dim>
Outflow<Dim>::Outflow(CkMigrateMessage* const msg) noexcept
    : BoundaryCondition<Dim>(msg) {}

template <size_t Dim>
std::optional<std::string> Outflow<Dim>::dg_outflow(
    const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
        face_mesh_velocity,
    const tnsr::i<DataVector, Dim>& normal_covector,
    const tnsr::I<DataVector, Dim>& /*normal_vector*/,
    const Scalar<DataVector>& gamma1, const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim>& shift) const noexcept {
  tnsr::a<DataVector, 3, Frame::Inertial> char_speeds{lapse.size()};
  characteristic_speeds(make_not_null(&char_speeds), gamma1, lapse, shift,
                        normal_covector);

  if (face_mesh_velocity.has_value()) {
    const auto face_speed = dot_product(normal_covector, *face_mesh_velocity);
    for (auto& char_speed : char_speeds) {
      char_speed -= get(face_speed);
    }
  }
  for (size_t i = 0; i < char_speeds.size(); ++i) {
    if (min(char_speeds[i]) < 0.) {
      return MakeString{}
             << "Detected negative characteristic speed at boundary with "
                "outflowing boundary conditions specified. The speed is "
             << min(char_speeds[i]) << " for index " << i
             << ". To see which characteristic field this corresponds to, "
                "check the function `characteristic_speeds` in "
                "Evolution/Systems/CurvedScalarWave/Characteristics.hpp.";
    }
  }
  return std::nullopt;  // LCOV_EXCL_LINE
}

template <size_t Dim>
// NOLINTNEXTLINE
PUP::able::PUP_ID Outflow<Dim>::my_PUP_ID = 0;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data) template class Outflow<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace CurvedScalarWave::BoundaryConditions
