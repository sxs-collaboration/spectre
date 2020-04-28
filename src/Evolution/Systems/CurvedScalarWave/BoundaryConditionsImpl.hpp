// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/DataOnSlice.hpp"
#include "DataStructures/LeviCivitaIterator.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/IndexToSliceAt.hpp"
#include "Domain/Tags.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Evolution/Systems/CurvedScalarWave/Characteristics.hpp"
#include "Evolution/Systems/CurvedScalarWave/Constraints.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits.hpp"

/// \cond
namespace Tags {
template <typename Tag>
struct Magnitude;
}  // namespace Tags
// IWYU pragma: no_forward_declare db::DataBox
/// \endcond

// debugPK
constexpr bool debugPK = false;

namespace CurvedScalarWave {
namespace Actions {

namespace BoundaryConditions_detail {}  // namespace BoundaryConditions_detail

namespace BoundaryConditions_detail {
enum class VPsiBcMethod { Freezing, ConstraintPreservingBjorhus, Unknown };
enum class VZeroBcMethod {
  Freezing,
  ConstraintPreservingBjorhus,
  // The condition below is borrowed from SpEC, where it is used as the
  // BC on VZero when "Constraint" is the requested BC type
  ConstraintPreservingPenalty,
  Unknown
};
enum class VPlusBcMethod { Freezing, Unknown };
enum class VMinusBcMethod { Freezing, ConstraintPreservingBjorhus, Unknown };

// Function to compute the smallest positive (or largest negative)
// characteristic speed
template <size_t VolumeDim>
double min_characteristic_speed(
    const typename CurvedScalarWave::Tags::CharacteristicSpeeds<
        VolumeDim>::type& char_speeds) noexcept {
  std::array<double, 4> min_speeds{
      {min(char_speeds.at(0)), min(char_speeds.at(1)), min(char_speeds.at(2)),
       min(char_speeds.at(3))}};
  return *std::min_element(min_speeds.begin(), min_speeds.end());
}

// Function to apply the newly computed value for evolution RHS
// wherever the particular characteristic field is incoming, and leaving
// the RHS unchanged when the same is NOT incoming (including for char speed 0).
template <typename T, typename DataType>
T set_bc_when_char_speed_is_negative(const T& rhs_char_dt_u,
                                     const T& desired_bc_dt_u,
                                     const DataType& char_speed_u) noexcept {
  // debugPK
  bool char_speed_always_positive = true;
  auto bc_dt_u = rhs_char_dt_u;
  auto it1 = bc_dt_u.begin();
  auto it2 = desired_bc_dt_u.begin();
  for (; it2 != desired_bc_dt_u.end(); ++it1, ++it2) {
    for (size_t i = 0; i < it1->size(); ++i) {
      if (char_speed_u[i] < 0.) {
        (*it1)[i] = (*it2)[i];
        char_speed_always_positive = false;  // debugPK
      }
    }
  }
  // debugPK
  if (char_speed_always_positive and debugPK) {
    Parallel::printf("Char speeds NEVER NEGATIVE..!\n");
  }
  return bc_dt_u;
}

template <size_t VolumeDim>
using all_local_vars = tmpl::list<
    // lapse, shift, and derivatives of lapse, shift
    gr::Tags::Lapse<DataVector>,
    gr::Tags::Shift<VolumeDim, Frame::Inertial, DataVector>,
    ::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<VolumeDim>,
                  Frame::Inertial>,
    ::Tags::deriv<gr::Tags::Shift<VolumeDim, Frame::Inertial, DataVector>,
                  tmpl::size_t<VolumeDim>, Frame::Inertial>,
    // Interface normal vector
    ::Tags::TempI<0, VolumeDim, Frame::Inertial, DataVector>,
    // Char speeds
    ::Tags::TempScalar<1, DataVector>, ::Tags::TempScalar<2, DataVector>,
    ::Tags::TempScalar<3, DataVector>, ::Tags::TempScalar<4, DataVector>,
    // C_{jk}
    Tags::TwoIndexConstraint<VolumeDim>,
    // Characteristic projected time derivatives of evolved
    // fields
    ::Tags::TempScalar<6, DataVector>,
    ::Tags::Tempi<7, VolumeDim, Frame::Inertial, DataVector>,
    ::Tags::TempScalar<8, DataVector>, ::Tags::TempScalar<9, DataVector>,
    // Constraint damping parameter gamma2
    Tags::ConstraintGamma2,
    // derivatives of pi, phi
    ::Tags::deriv<Pi, tmpl::size_t<VolumeDim>, Frame::Inertial>,
    ::Tags::deriv<Phi<VolumeDim>, tmpl::size_t<VolumeDim>, Frame::Inertial>,
    // Preallocated memory to store boundary conditions
    ::Tags::TempScalar<12, DataVector>,
    ::Tags::Tempi<13, VolumeDim, Frame::Inertial, DataVector>,
    ::Tags::TempScalar<14, DataVector>, ::Tags::TempScalar<15, DataVector>>;

// \brief This function computes intermediate variables needed for
// Bjorhus-type constraint preserving boundary conditions for the
// CurvedScalarWave system
template <size_t VolumeDim, typename TagsList, typename DbTags,
          typename VarsTagsList, typename DtVarsTagsList>
void local_variables(
    gsl::not_null<TempBuffer<TagsList>*> buffer, const db::DataBox<DbTags>& box,
    const Direction<VolumeDim>& direction, const size_t& dimension,
    const typename domain::Tags::Mesh<VolumeDim>::type& mesh,
    const Variables<VarsTagsList>& /* vars */,
    const Variables<DtVarsTagsList>& dt_vars,
    const tnsr::i<DataVector, VolumeDim, Frame::Inertial>&
        unit_interface_normal_one_form,
    const typename Tags::CharacteristicSpeeds<VolumeDim>::type& char_speeds,
    const Scalar<DataVector>& constraint_gamma2) noexcept {
  // Extract quantities from databox that are needed to compute
  // intermediate variables
  using tags_needed_on_slice = tmpl::list<
      // lapse, shift, and inverse_spatial_metric
      gr::Tags::Lapse<DataVector>,
      gr::Tags::Shift<VolumeDim, Frame::Inertial, DataVector>,
      gr::Tags::InverseSpatialMetric<VolumeDim, Frame::Inertial, DataVector>,
      // derivatives of lapse, shift
      ::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<VolumeDim>,
                    Frame::Inertial>,
      ::Tags::deriv<gr::Tags::Shift<VolumeDim, Frame::Inertial, DataVector>,
                    tmpl::size_t<VolumeDim>, Frame::Inertial>,
      // derivatives of pi and phi
      ::Tags::deriv<Pi, tmpl::size_t<VolumeDim>, Frame::Inertial>,
      ::Tags::deriv<Phi<VolumeDim>, tmpl::size_t<VolumeDim>, Frame::Inertial>,
      Tags::TwoIndexConstraint<VolumeDim>>;
  const auto vars_on_this_slice = db::data_on_slice(
      box, mesh.extents(), dimension,
      index_to_slice_at(mesh.extents(), direction), tags_needed_on_slice{});
  // -1. spacetime quantities
  get<gr::Tags::Lapse<DataVector>>(*buffer) =
      get<gr::Tags::Lapse<DataVector>>(vars_on_this_slice);
  get<gr::Tags::Shift<VolumeDim, Frame::Inertial, DataVector>>(*buffer) =
      get<gr::Tags::Shift<VolumeDim, Frame::Inertial, DataVector>>(
          vars_on_this_slice);
  get<::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<VolumeDim>,
                    Frame::Inertial>>(*buffer) =
      get<::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<VolumeDim>,
                        Frame::Inertial>>(vars_on_this_slice);
  get<::Tags::deriv<gr::Tags::Shift<VolumeDim, Frame::Inertial, DataVector>,
                    tmpl::size_t<VolumeDim>, Frame::Inertial>>(*buffer) =
      get<::Tags::deriv<gr::Tags::Shift<VolumeDim, Frame::Inertial, DataVector>,
                        tmpl::size_t<VolumeDim>, Frame::Inertial>>(
          vars_on_this_slice);

  // 0. interface normal vector
  const auto& inverse_spatial_metric = get<
      gr::Tags::InverseSpatialMetric<VolumeDim, Frame::Inertial, DataVector>>(
      vars_on_this_slice);
  get<::Tags::TempI<0, VolumeDim, Frame::Inertial, DataVector>>(*buffer) =
      raise_or_lower_index(unit_interface_normal_one_form,
                           inverse_spatial_metric);

  // 1-4. Characteristic speeds
  get(get<::Tags::TempScalar<1, DataVector>>(*buffer)) = char_speeds.at(0);
  get(get<::Tags::TempScalar<2, DataVector>>(*buffer)) = char_speeds.at(1);
  get(get<::Tags::TempScalar<3, DataVector>>(*buffer)) = char_speeds.at(2);
  get(get<::Tags::TempScalar<4, DataVector>>(*buffer)) = char_speeds.at(3);

  // 5. C_{ij}
  get<Tags::TwoIndexConstraint<VolumeDim>>(*buffer) =
      get<Tags::TwoIndexConstraint<VolumeDim>>(vars_on_this_slice);

  // 6-9. Characteristic projected time derivatives of evolved fields
  // storage for DT<UChar> = CharProjection(dt<U>)
  const auto& rhs_dt_psi = get<::Tags::dt<Psi>>(dt_vars);
  const auto& rhs_dt_pi = get<::Tags::dt<Pi>>(dt_vars);
  const auto& rhs_dt_phi = get<::Tags::dt<Phi<VolumeDim>>>(dt_vars);
  const auto char_projected_dt_u = characteristic_fields(
      constraint_gamma2, inverse_spatial_metric, rhs_dt_psi, rhs_dt_pi,
      rhs_dt_phi, unit_interface_normal_one_form);
  get<::Tags::TempScalar<6, DataVector>>(*buffer) =
      get<Tags::VPsi>(char_projected_dt_u);
  get<::Tags::Tempi<7, VolumeDim, Frame::Inertial, DataVector>>(*buffer) =
      get<Tags::VZero<VolumeDim>>(char_projected_dt_u);
  get<::Tags::TempScalar<8, DataVector>>(*buffer) =
      get<Tags::VPlus>(char_projected_dt_u);
  get<::Tags::TempScalar<9, DataVector>>(*buffer) =
      get<Tags::VMinus>(char_projected_dt_u);

  // 10. Constraint damping parameter
  get<Tags::ConstraintGamma2>(*buffer) = constraint_gamma2;

  // 11. Spatial derivatives of evolved variables: Pi
  get<::Tags::deriv<Pi, tmpl::size_t<VolumeDim>, Frame::Inertial>>(*buffer) =
      get<::Tags::deriv<Pi, tmpl::size_t<VolumeDim>, Frame::Inertial>>(
          vars_on_this_slice);
  get<::Tags::deriv<Phi<VolumeDim>, tmpl::size_t<VolumeDim>, Frame::Inertial>>(
      *buffer) = get<::Tags::deriv<Phi<VolumeDim>, tmpl::size_t<VolumeDim>,
                                   Frame::Inertial>>(vars_on_this_slice);
}

// \brief This struct sets boundary condition on dt<VPsi>
template <size_t VolumeDim>
struct set_dt_u_psi {
 private:
  using ReturnType = typename Tags::VPsi::type;

 public:
  template <typename TagsList, typename VarsTagsList, typename DtVarsTagsList>
  static ReturnType apply(const VPsiBcMethod Method,
                          TempBuffer<TagsList>& buffer,
                          const Variables<VarsTagsList>& vars,
                          const Variables<DtVarsTagsList>& /* dt_vars */,
                          const tnsr::i<DataVector, VolumeDim, Frame::Inertial>&
                          /* unit_normal_one_form */) noexcept {
    const auto& pi = get<Pi>(vars);
    const auto& phi = get<Phi<VolumeDim>>(vars);
    const auto& lapse = get<gr::Tags::Lapse<DataVector>>(buffer);
    const auto& shift =
        get<gr::Tags::Shift<VolumeDim, Frame::Inertial, DataVector>>(buffer);

    // Memory allocated for return type
    ReturnType& bc_dt_u_psi = get<::Tags::TempScalar<12, DataVector>>(buffer);

    std::fill(bc_dt_u_psi.begin(), bc_dt_u_psi.end(), 0.);
    // Switch on prescribed boundary condition method
    switch (Method) {
      case VPsiBcMethod::Freezing:
        return bc_dt_u_psi;
      case VPsiBcMethod::ConstraintPreservingBjorhus:
        return apply_bjorhus_constraint_preserving(make_not_null(&bc_dt_u_psi),
                                                   lapse, shift, phi, pi);
      case VPsiBcMethod::Unknown:
      default:
        ASSERT(false, "Requested BC method for VPsi not implemented!");
    }

    // dummy return to suppress compiler warning
    return ReturnType{};
  }

 private:
  static ReturnType apply_bjorhus_constraint_preserving(
      const gsl::not_null<ReturnType*> bc_dt_u_psi,
      const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, VolumeDim, Frame::Inertial>& shift,
      const tnsr::i<DataVector, VolumeDim, Frame::Inertial>& phi,
      const Scalar<DataVector>& pi) noexcept {
    ASSERT(get_size(get(*bc_dt_u_psi)) == get_size(get(pi)),
           "Size of input variables and temporary memory do not match.");
    const auto shift_dot_phi = dot_product(shift, phi);
    get(*bc_dt_u_psi) = get(shift_dot_phi) - get(lapse) * get(pi);
    return *bc_dt_u_psi;
  }
};

// \brief This struct sets boundary condition on dt<VZero>
template <size_t VolumeDim>
struct set_dt_u_zero {
 private:
  using ReturnType = typename Tags::VZero<VolumeDim>::type;

 public:
  template <typename TagsList, typename VarsTagsList, typename DtVarsTagsList>
  static ReturnType apply(const VZeroBcMethod Method,
                          TempBuffer<TagsList>& buffer,
                          const Variables<VarsTagsList>& vars,
                          const Variables<DtVarsTagsList>& /* dt_vars */,
                          const tnsr::i<DataVector, VolumeDim, Frame::Inertial>&
                              unit_normal_one_form) noexcept {
    // Not using auto below to enforce a loose test on the quantity being
    // fetched from the buffer
    const auto& deriv_lapse =
        get<::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<VolumeDim>,
                          Frame::Inertial>>(buffer);
    const auto& deriv_shift = get<
        ::Tags::deriv<gr::Tags::Shift<VolumeDim, Frame::Inertial, DataVector>,
                      tmpl::size_t<VolumeDim>, Frame::Inertial>>(buffer);
    const auto& deriv_pi =
        get<::Tags::deriv<Pi, tmpl::size_t<VolumeDim>, Frame::Inertial>>(
            buffer);
    const auto& deriv_phi =
        get<::Tags::deriv<Phi<VolumeDim>, tmpl::size_t<VolumeDim>,
                          Frame::Inertial>>(buffer);

    const auto& pi = get<Pi>(vars);
    const auto& phi = get<Phi<VolumeDim>>(vars);
    const auto& lapse = get<gr::Tags::Lapse<DataVector>>(buffer);
    const auto& shift =
        get<gr::Tags::Shift<VolumeDim, Frame::Inertial, DataVector>>(buffer);

    const auto& unit_interface_normal_vector =
        get<::Tags::TempI<0, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    const auto& two_index_constraint =
        get<Tags::TwoIndexConstraint<VolumeDim>>(buffer);
    const auto& char_projected_rhs_dt_u_zero =
        get<::Tags::Tempi<7, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    const typename Tags::CharacteristicSpeeds<VolumeDim>::type char_speeds{
        {get(get<::Tags::TempScalar<1, DataVector>>(buffer)),
         get(get<::Tags::TempScalar<2, DataVector>>(buffer)),
         get(get<::Tags::TempScalar<3, DataVector>>(buffer)),
         get(get<::Tags::TempScalar<4, DataVector>>(buffer))}};

    // Memory allocated for return type
    ReturnType& bc_dt_u_zero =
        get<::Tags::Tempi<13, VolumeDim, Frame::Inertial, DataVector>>(buffer);

    std::fill(bc_dt_u_zero.begin(), bc_dt_u_zero.end(), 0.);
    // Switch on prescribed boundary condition method
    switch (Method) {
      case VZeroBcMethod::Freezing:
        return bc_dt_u_zero;
      case VZeroBcMethod::ConstraintPreservingPenalty:
        return apply_penalty_constraint_preserving(
            make_not_null(&bc_dt_u_zero), unit_interface_normal_vector,
            two_index_constraint, char_projected_rhs_dt_u_zero, char_speeds);
      case VZeroBcMethod::ConstraintPreservingBjorhus:
        return apply_bjorhus_constraint_preserving(
            make_not_null(&bc_dt_u_zero), deriv_lapse, deriv_shift, deriv_phi,
            deriv_pi, lapse, shift, phi, pi, unit_interface_normal_vector,
            unit_normal_one_form);
      case VZeroBcMethod::Unknown:
      default:
        ASSERT(false, "Requested BC method fo VZero not implemented!");
    }
    // dummy return to suppress compiler warning
    return ReturnType{};
  }

 private:
  // The condition below is borrowed from SpEC, where it is used as the
  // BC on VZero when "Constraint" is the requested BC type. It applies
  // penalty-type correction to the RHS for dt<VZero>.
  static ReturnType apply_penalty_constraint_preserving(
      const gsl::not_null<ReturnType*> bc_dt_u_zero,
      const tnsr::I<DataVector, VolumeDim, Frame::Inertial>&
          unit_interface_normal_vector,
      const tnsr::ij<DataVector, VolumeDim, Frame::Inertial>&
          two_index_constraint,
      const tnsr::i<DataVector, VolumeDim, Frame::Inertial>&
          char_projected_rhs_dt_u_zero,
      const std::array<DataVector, 4>& char_speeds) noexcept {
    ASSERT(get_size(get<0>(*bc_dt_u_zero)) ==
               get_size(get<0>(unit_interface_normal_vector)),
           "Size of input variables and temporary memory do not match.");

    for (size_t i = 0; i < VolumeDim; ++i) {
      bc_dt_u_zero->get(i) = char_projected_rhs_dt_u_zero.get(i);
      for (size_t j = 0; j < VolumeDim; ++j) {
        // Note: char_speed<VZero> should be identically 0!
        bc_dt_u_zero->get(i) += char_speeds.at(1) *
                                unit_interface_normal_vector.get(j) *
                                two_index_constraint.get(j, i);
      }
    }
    return *bc_dt_u_zero;
  }

  // According to Holst+ 2004, this condition for VZero also depends on
  // whether VPsi is incoming or not. SpEC only uses the expression valid for
  // the case when VPsi is incoming...!
  static ReturnType apply_bjorhus_constraint_preserving(
      const gsl::not_null<ReturnType*> bc_dt_u_zero,
      const tnsr::i<DataVector, VolumeDim, Frame::Inertial>& deriv_lapse,
      const tnsr::iJ<DataVector, VolumeDim, Frame::Inertial>& deriv_shift,
      const tnsr::ij<DataVector, VolumeDim, Frame::Inertial>& deriv_phi,
      const tnsr::i<DataVector, VolumeDim, Frame::Inertial>& deriv_pi,
      const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, VolumeDim, Frame::Inertial>& shift,
      const tnsr::i<DataVector, VolumeDim, Frame::Inertial>& phi,
      const Scalar<DataVector>& pi,
      const tnsr::I<DataVector, VolumeDim, Frame::Inertial>&
          unit_interface_normal_vector,
      const tnsr::i<DataVector, VolumeDim, Frame::Inertial>&
          unit_interface_normal_one_form) noexcept {
    ASSERT(get_size(get<0>(*bc_dt_u_zero)) == get_size(get(pi)),
           "Size of input variables and temporary memory do not match.");

    auto tmp = make_with_value<tnsr::i<DataVector, VolumeDim, Frame::Inertial>>(
        pi, 0.);
    for (size_t j = 0; j < VolumeDim; ++j) {
      tmp.get(j) = -get(lapse) * deriv_pi.get(j) - get(pi) * deriv_lapse.get(j);
      for (size_t k = 0; k < VolumeDim; ++k) {
        tmp.get(j) += deriv_shift.get(j, k) * phi.get(k) +
                      shift.get(k) * deriv_phi.get(j, k);
      }
    }

    for (size_t i = 0; i < VolumeDim; ++i) {
      bc_dt_u_zero->get(i) = tmp.get(i);
      for (size_t j = 0; j < VolumeDim; ++j) {
        bc_dt_u_zero->get(i) -= unit_interface_normal_one_form.get(i) *
                                unit_interface_normal_vector.get(j) *
                                tmp.get(j);
      }
    }

    return *bc_dt_u_zero;
  }
};

// \brief This struct sets boundary condition on dt<VMinus>
//
// \note Before calling this struct, one must store the final BC values for
// dt<VPsi>, i.e. after accounting for whether VPsi is actually incoming or not,
// in the buffer. This condition needs it.
// This struct must therefore be applied *after* applying `set_dt_u_psi`.
template <size_t VolumeDim>
struct set_dt_u_minus {
 private:
  using ReturnType = typename Tags::VMinus::type;

 public:
  template <typename TagsList, typename VarsTagsList, typename DtVarsTagsList>
  static ReturnType apply(const VMinusBcMethod Method,
                          TempBuffer<TagsList>& buffer,
                          const Variables<VarsTagsList>& /* vars */,
                          const Variables<DtVarsTagsList>& /* dt_vars */,
                          const tnsr::i<DataVector, VolumeDim, Frame::Inertial>&
                          /* unit_normal_one_form */) noexcept {
    const auto& constraint_gamma2 = get<Tags::ConstraintGamma2>(buffer);
    // Note that `bc_dt_u_psi` is the *final* value of dt<VPsi> that has been
    // set after considering the sign of char speeds of the char field VPsi
    const auto& bc_dt_u_psi = get<::Tags::TempScalar<6, DataVector>>(buffer);

    // Memory allocated for return type
    ReturnType& bc_dt_u_minus = get<::Tags::TempScalar<15, DataVector>>(buffer);
    std::fill(bc_dt_u_minus.begin(), bc_dt_u_minus.end(), 0.);
    // Switch on prescribed boundary condition method
    switch (Method) {
      case VMinusBcMethod::Freezing:
        return bc_dt_u_minus;
      case VMinusBcMethod::ConstraintPreservingBjorhus:
        return apply_bjorhus_constraint_preserving(
            make_not_null(&bc_dt_u_minus), constraint_gamma2, bc_dt_u_psi);
      case VMinusBcMethod::Unknown:
      default:
        ASSERT(false, "Requested BC method for VMinus is not implemented!");
    }
    // dummy return to suppress compiler warning
    return ReturnType{};
  }

 private:
  static ReturnType apply_bjorhus_constraint_preserving(
      const gsl::not_null<ReturnType*> bc_dt_u_minus,
      const Scalar<DataVector>& constraint_gamma2,
      const Scalar<DataVector>& bc_dt_u_psi) noexcept {
    ASSERT(get_size(get(*bc_dt_u_minus)) == get_size(get(constraint_gamma2)),
           "Size of input variables and temporary memory do not match.");
    get(*bc_dt_u_minus) = -get(constraint_gamma2) * get(bc_dt_u_psi);
    return *bc_dt_u_minus;
  }
};

// \brief This struct sets boundary condition on dt<VPlus>
template <size_t VolumeDim>
struct set_dt_u_plus {
 private:
  using ReturnType = typename Tags::VPlus::type;

 public:
  template <typename TagsList, typename VarsTagsList, typename DtVarsTagsList>
  static ReturnType apply(const VPlusBcMethod Method,
                          TempBuffer<TagsList>& buffer,
                          const Variables<VarsTagsList>& /* vars */,
                          const Variables<DtVarsTagsList>& /* dt_vars */,
                          const tnsr::i<DataVector, VolumeDim, Frame::Inertial>&
                          /* unit_normal_one_form */) noexcept {
    // Memory allocated for return type
    ReturnType& bc_dt_u_plus = get<::Tags::TempScalar<14, DataVector>>(buffer);
    std::fill(bc_dt_u_plus.begin(), bc_dt_u_plus.end(), 0.);
    // Switch on prescribed boundary condition method
    switch (Method) {
      case VPlusBcMethod::Freezing:
        return bc_dt_u_plus;
      case VPlusBcMethod::Unknown:
      default:
        ASSERT(false, "Requested BC method fo VPlus is not implemented!");
    }
    // dummy return to suppress compiler warning
    return ReturnType{};
  }

 private:
};

}  // namespace BoundaryConditions_detail
}  // namespace Actions
}  // namespace CurvedScalarWave
