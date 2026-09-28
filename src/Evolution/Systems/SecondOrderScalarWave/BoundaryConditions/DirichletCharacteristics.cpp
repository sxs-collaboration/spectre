// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/SecondOrderScalarWave/BoundaryConditions/DirichletCharacteristics.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <pup.h>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/SecondOrderScalarWave/Characteristics.hpp"
#include "Evolution/Systems/SecondOrderScalarWave/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/Factory.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"  // IWYU pragma: keep

namespace SecondOrderScalarWave::BoundaryConditions {
template <size_t Dim>
DirichletCharacteristics<Dim>::DirichletCharacteristics(
    const DirichletCharacteristics& rhs)
    : BoundaryCondition<Dim>{dynamic_cast<const BoundaryCondition<Dim>&>(rhs)},
      analytic_prescription_(rhs.analytic_prescription_ != nullptr
                                 ? rhs.analytic_prescription_->get_clone()
                                 : nullptr),
      copy_psi_from_interior_(rhs.copy_psi_from_interior_) {}

template <size_t Dim>
DirichletCharacteristics<Dim>& DirichletCharacteristics<Dim>::operator=(
    const DirichletCharacteristics& rhs) {
  if (&rhs == this) {
    return *this;
  }
  analytic_prescription_ = rhs.analytic_prescription_ != nullptr
                               ? rhs.analytic_prescription_->get_clone()
                               : nullptr;
  copy_psi_from_interior_ = rhs.copy_psi_from_interior_;
  return *this;
}

template <size_t Dim>
DirichletCharacteristics<Dim>::DirichletCharacteristics(
    CkMigrateMessage* const msg)
    : BoundaryCondition<Dim>(msg) {}

template <size_t Dim>
DirichletCharacteristics<Dim>::DirichletCharacteristics(
    std::optional<std::unique_ptr<evolution::initial_data::InitialData>>
        analytic_prescription,
    const bool copy_psi_from_interior)
    : analytic_prescription_(
          std::move(analytic_prescription).value_or(nullptr)),
      copy_psi_from_interior_(copy_psi_from_interior) {}

template <size_t Dim>
std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
DirichletCharacteristics<Dim>::get_clone() const {
  return std::make_unique<DirichletCharacteristics>(*this);
}

template <size_t Dim>
void DirichletCharacteristics<Dim>::pup(PUP::er& p) {
  BoundaryCondition<Dim>::pup(p);
  p | analytic_prescription_;
  p | copy_psi_from_interior_;
}

namespace {
// The incoming characteristic mode v^- on the ghost side: zero if there is no
// `analytic_prescription` (ZeroIncomingMode), otherwise v^- of the analytic
// prescription.
template <size_t Dim>
Scalar<DataVector> ghost_v_minus(
    const evolution::initial_data::InitialData* const analytic_prescription,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& coords,
    const double time) {
  if (analytic_prescription == nullptr) {
    return Scalar<DataVector>{get<0>(normal_covector).size(), 0.0};
  }
  using tags = tmpl::list<Tags::Psi, Tags::Pi, Tags::Phi<Dim>>;
  const auto analytic =
      call_with_dynamic_type<tuples::tagged_tuple_from_typelist<tags>,
                             Solutions::all_solutions<Dim>>(
          analytic_prescription, [&coords, &time](const auto* const solution) {
            return solution->variables(coords, time, tags{});
          });
  const auto analytic_char_fields = characteristic_fields(
      get<Tags::Pi>(analytic), get<Tags::Phi<Dim>>(analytic), normal_covector);
  return get<Tags::VMinus>(analytic_char_fields);
}
}  // namespace

template <size_t Dim>
std::optional<std::string> DirichletCharacteristics<Dim>::dg_ghost(
    const gsl::not_null<Scalar<DataVector>*> psi,
    const gsl::not_null<Scalar<DataVector>*> pi,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*> phi,
    const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
        face_mesh_velocity,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
    const Scalar<DataVector>& boundary_psi,
    const Scalar<DataVector>& interior_psi,
    const Scalar<DataVector>& interior_pi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& interior_phi,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& coords,
    const double time) const {
  if (face_mesh_velocity.has_value()) {
    return "DirichletCharacteristics does not support moving meshes for now.";
  }
  // Outgoing v^+ and zero-speed v^0 come from the interior; the incoming v^-
  // comes from the analytic data (or is zero).
  const auto interior_char_fields =
      characteristic_fields(interior_pi, interior_phi, normal_covector);
  const auto v_minus = ghost_v_minus<Dim>(analytic_prescription_.get(),
                                          normal_covector, coords, time);
  const auto ghost_fields = fields_from_inverse_characteristic_transform(
      get<Tags::VZero<Dim>>(interior_char_fields),
      get<Tags::VPlus>(interior_char_fields), v_minus, normal_covector);

  *psi = copy_psi_from_interior_ ? interior_psi : boundary_psi;
  *pi = get<Tags::Pi>(ghost_fields);
  *phi = get<Tags::Phi<Dim>>(ghost_fields);

  return std::nullopt;
}

template <size_t Dim>
std::optional<std::string>
DirichletCharacteristics<Dim>::boundary_field_time_derivatives(
    const gsl::not_null<Scalar<DataVector>*> dt_boundary_psi,
    const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
        face_mesh_velocity,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
    const Scalar<DataVector>& /*boundary_psi*/,
    const Scalar<DataVector>& interior_pi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& interior_phi,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& coords,
    const double time) const {
  if (face_mesh_velocity.has_value()) {
    return "DirichletCharacteristics does not support moving meshes for now.";
  }
  // dt BoundaryPsi = -Pi_boundary = -0.5 * (v^+ + v^-).
  // The boundary value is integrated regardless of CopyPsiFromInterior.
  const auto interior_char_fields =
      characteristic_fields(interior_pi, interior_phi, normal_covector);
  const auto v_minus = ghost_v_minus<Dim>(analytic_prescription_.get(),
                                          normal_covector, coords, time);
  get(*dt_boundary_psi) =
      -0.5 * (get(get<Tags::VPlus>(interior_char_fields)) + get(v_minus));
  return std::nullopt;
}

template <size_t Dim>
// NOLINTNEXTLINE
PUP::able::PUP_ID DirichletCharacteristics<Dim>::my_PUP_ID = 0;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data) \
  template class DirichletCharacteristics<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace SecondOrderScalarWave::BoundaryConditions
