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
#include "DataStructures/SliceVariables.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/IndexToSliceAt.hpp"
#include "Domain/Tags.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Evolution/Systems/ScalarWave/Characteristics.hpp"
#include "Evolution/Systems/ScalarWave/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/FluxCommunicationTypes.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Printf.hpp"
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

namespace ScalarWave {
namespace Actions {

namespace BoundaryConditions_detail {}  // namespace BoundaryConditions_detail

namespace BoundaryConditions_detail {
enum class VPsiBcMethod { Freezing, ConstraintPreservingBjorhus, Unknown };
enum class VZeroBcMethod {
  Freezing,
  ConstraintPreservingBjorhus,
  ConstraintPreservingDirichlet,
  Unknown
};
enum class VPlusBcMethod { Freezing, Unknown };
enum class VMinusBcMethod {
  Freezing,
  ConstraintPreservingBjorhus,
  DampingVMinusBjorhus,
  Unknown
};

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
    // Interface normal vector
    ::Tags::TempI<0, VolumeDim, Frame::Inertial, DataVector>,
    // Char speeds
    ::Tags::TempScalar<1, DataVector>, ::Tags::TempScalar<2, DataVector>,
    ::Tags::TempScalar<3, DataVector>, ::Tags::TempScalar<4, DataVector>,
    // C_{jk}
    ::Tags::Tempij<5, VolumeDim, Frame::Inertial, DataVector>,
    // Characteristic projected time derivatives of evolved
    // fields
    ::Tags::TempScalar<6, DataVector>,
    ::Tags::Tempi<7, VolumeDim, Frame::Inertial, DataVector>,
    ::Tags::TempScalar<8, DataVector>, ::Tags::TempScalar<9, DataVector>,
    // Constraint damping parameter gamma2
    ::Tags::TempScalar<10, DataVector>,
    // derivatives of pi
    ::Tags::Tempi<11, VolumeDim, Frame::Inertial, DataVector>,
    // Preallocated memory to store boundary conditions
    ::Tags::TempScalar<12, DataVector>,
    ::Tags::Tempi<13, VolumeDim, Frame::Inertial, DataVector>,
    ::Tags::TempScalar<14, DataVector>, ::Tags::TempScalar<15, DataVector>>;

// \brief This function computes intermediate variables needed for
// Bjorhus-type constraint preserving boundary conditions for the
// ScalarWave system
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
      ::Tags::deriv<Pi, tmpl::size_t<VolumeDim>, Frame::Inertial>,
      /*Tags::ConstraintGamma2,*/ Tags::TwoIndexConstraint<VolumeDim>>;
  const auto vars_on_this_slice = db::data_on_slice(
      box, mesh.extents(), dimension,
      index_to_slice_at(mesh.extents(), direction), tags_needed_on_slice{});

  // 0. interface normal vector
  auto& local_interface_normal_vector =
      get<::Tags::TempI<0, VolumeDim, Frame::Inertial, DataVector>>(*buffer);
  for (size_t i = 0; i < VolumeDim; ++i) {
    local_interface_normal_vector.get(i) =
        unit_interface_normal_one_form.get(i);
  }
  // 1-4. Characteristic speeds
  get(get<::Tags::TempScalar<1, DataVector>>(*buffer)) = char_speeds.at(0);
  get(get<::Tags::TempScalar<2, DataVector>>(*buffer)) = char_speeds.at(1);
  get(get<::Tags::TempScalar<3, DataVector>>(*buffer)) = char_speeds.at(2);
  get(get<::Tags::TempScalar<4, DataVector>>(*buffer)) = char_speeds.at(3);
  // 5. C_{ij}
  get<::Tags::Tempij<5, VolumeDim, Frame::Inertial, DataVector>>(*buffer) =
      get<Tags::TwoIndexConstraint<VolumeDim>>(vars_on_this_slice);
  // 6-9. Characteristic projected time derivatives of evolved fields
  // storage for DT<UChar> = CharProjection(dt<U>)
  const auto& rhs_dt_psi = get<::Tags::dt<Psi>>(dt_vars);
  const auto& rhs_dt_pi = get<::Tags::dt<Pi>>(dt_vars);
  const auto& rhs_dt_phi = get<::Tags::dt<Phi<VolumeDim>>>(dt_vars);
  const auto char_projected_dt_u =
      characteristic_fields(constraint_gamma2, rhs_dt_psi, rhs_dt_pi,
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
  get<::Tags::TempScalar<10, DataVector>>(*buffer) = constraint_gamma2;
  // 11. Spatial derivatives of evolved variables: Pi
  get<::Tags::Tempi<11, VolumeDim, Frame::Inertial, DataVector>>(*buffer) =
      get<::Tags::deriv<Pi, tmpl::size_t<VolumeDim>, Frame::Inertial>>(
          vars_on_this_slice);
}

// \brief This struct sets boundary condition on dt<VPsi>
template <typename ReturnType, size_t VolumeDim>
struct set_dt_v_psi {
  template <typename TagsList, typename VarsTagsList, typename DtVarsTagsList>
  static ReturnType apply(const VPsiBcMethod Method,
                          TempBuffer<TagsList>& buffer,
                          const Variables<VarsTagsList>& vars,
                          const Variables<DtVarsTagsList>& /* dt_vars */,
                          const tnsr::i<DataVector, VolumeDim, Frame::Inertial>&
                          /* unit_normal_one_form */) noexcept {
    const typename Pi::type& pi = get<Pi>(vars);
    // Memory allocated for return type
    ReturnType& bc_dt_v_psi = get<::Tags::TempScalar<12, DataVector>>(buffer);
    std::fill(bc_dt_v_psi.begin(), bc_dt_v_psi.end(), 0.);
    // Switch on prescribed boundary condition method
    switch (Method) {
      case VPsiBcMethod::Freezing:
        return bc_dt_v_psi;
      case VPsiBcMethod::ConstraintPreservingBjorhus:
        return apply_bjorhus_constraint_preserving(make_not_null(&bc_dt_v_psi),
                                                   pi);
      case VPsiBcMethod::Unknown:
      default:
        ASSERT(false, "Requested BC method for VPsi not implemented!");
    }
    // dummy return to suppress compiler warning
    return ReturnType{};
  }

 private:
  static ReturnType apply_bjorhus_constraint_preserving(
      const gsl::not_null<ReturnType*> bc_dt_v_psi,
      const Scalar<DataVector>& pi) noexcept {
    ASSERT(get_size(get(*bc_dt_v_psi)) == get_size(get(pi)),
           "Size of input variables and temporary memory do not match.");
    get(*bc_dt_v_psi) = -get(pi);
    return *bc_dt_v_psi;
  }
};

// \brief This struct sets boundary condition on dt<VZero>
template <typename ReturnType, size_t VolumeDim>
struct set_dt_v_zero {
  template <typename TagsList, typename VarsTagsList, typename DtVarsTagsList>
  static ReturnType apply(const VZeroBcMethod Method,
                          TempBuffer<TagsList>& buffer,
                          const Variables<VarsTagsList>& /* vars */,
                          const Variables<DtVarsTagsList>& /* dt_vars */,
                          const tnsr::i<DataVector, VolumeDim, Frame::Inertial>&
                              unit_normal_one_form) noexcept {
    // Not using auto below to enforce a loose test on the quantity being
    // fetched from the buffer
    const tnsr::I<DataVector, VolumeDim,
                  Frame::Inertial>& unit_interface_normal_vector =
        get<::Tags::TempI<0, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    const typename Tags::TwoIndexConstraint<VolumeDim>::type&
        two_index_constraint =
            get<::Tags::Tempij<5, VolumeDim, Frame::Inertial, DataVector>>(
                buffer);
    const typename Tags::VZero<VolumeDim>::type& char_projected_rhs_dt_v_zero =
        get<::Tags::Tempi<7, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    const typename Tags::CharacteristicSpeeds<VolumeDim>::type char_speeds{
        {get(get<::Tags::TempScalar<1, DataVector>>(buffer)),
         get(get<::Tags::TempScalar<2, DataVector>>(buffer)),
         get(get<::Tags::TempScalar<3, DataVector>>(buffer)),
         get(get<::Tags::TempScalar<4, DataVector>>(buffer))}};
    const auto& d_pi =
        get<::Tags::Tempi<11, VolumeDim, Frame::Inertial, DataVector>>(buffer);

    // Memory allocated for return type
    ReturnType& bc_dt_v_zero =
        get<::Tags::Tempi<13, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    std::fill(bc_dt_v_zero.begin(), bc_dt_v_zero.end(), 0.);
    // Switch on prescribed boundary condition method
    switch (Method) {
      case VZeroBcMethod::Freezing:
        return bc_dt_v_zero;
      case VZeroBcMethod::ConstraintPreservingBjorhus:
        return apply_bjorhus_constraint_preserving(
            make_not_null(&bc_dt_v_zero), unit_interface_normal_vector,
            two_index_constraint, char_projected_rhs_dt_v_zero, char_speeds);
      case VZeroBcMethod::ConstraintPreservingDirichlet:
        return apply_dirichlet_constraint_preserving(
            make_not_null(&bc_dt_v_zero), unit_interface_normal_vector,
            unit_normal_one_form, d_pi);
      case VZeroBcMethod::Unknown:
      default:
        ASSERT(false, "Requested BC method fo VZero not implemented!");
    }
    // dummy return to suppress compiler warning
    return ReturnType{};
  }

 private:
  static ReturnType apply_bjorhus_constraint_preserving(
      const gsl::not_null<ReturnType*> bc_dt_v_zero,
      const tnsr::I<DataVector, VolumeDim, Frame::Inertial>&
          unit_interface_normal_vector,
      const tnsr::ij<DataVector, VolumeDim, Frame::Inertial>&
          two_index_constraint,
      const tnsr::i<DataVector, VolumeDim, Frame::Inertial>&
          char_projected_rhs_dt_v_zero,
      const std::array<DataVector, 4>& char_speeds) noexcept {
    ASSERT(get_size(get<0>(*bc_dt_v_zero)) ==
               get_size(get<0>(unit_interface_normal_vector)),
           "Size of input variables and temporary memory do not match.");
    for (size_t i = 0; i < VolumeDim; ++i) {
      bc_dt_v_zero->get(i) = char_projected_rhs_dt_v_zero.get(i);
      for (size_t j = 0; j < VolumeDim; ++j) {
        // Note: char_speed<VZero> should be identically 0!
        bc_dt_v_zero->get(i) += char_speeds.at(1) *
                                unit_interface_normal_vector.get(j) *
                                two_index_constraint.get(j, i);
      }
    }
    return *bc_dt_v_zero;
  }

  static ReturnType apply_dirichlet_constraint_preserving(
      const gsl::not_null<ReturnType*> bc_dt_v_zero,
      const tnsr::I<DataVector, VolumeDim, Frame::Inertial>&
          unit_interface_normal_vector,
      const tnsr::i<DataVector, VolumeDim, Frame::Inertial>&
          unit_interface_normal_one_form,
      const tnsr::i<DataVector, VolumeDim, Frame::Inertial>& d_pi) noexcept {
    ASSERT(get_size(get<0>(*bc_dt_v_zero)) == get_size(get<0>(d_pi)),
           "Size of input variables and temporary memory do not match.");
    for (size_t i = 0; i < VolumeDim; ++i) {
      bc_dt_v_zero->get(i) = -d_pi.get(i);
      for (size_t j = 0; j < VolumeDim; ++j) {
        bc_dt_v_zero->get(i) += unit_interface_normal_one_form.get(i) *
                                unit_interface_normal_vector.get(j) *
                                d_pi.get(j);
      }
    }
    return *bc_dt_v_zero;
  }
};

// \brief This struct sets boundary condition on dt<VPlus>
template <typename ReturnType, size_t VolumeDim>
struct set_dt_v_plus {
  template <typename TagsList, typename VarsTagsList, typename DtVarsTagsList>
  static ReturnType apply(const VPlusBcMethod Method,
                          TempBuffer<TagsList>& buffer,
                          const Variables<VarsTagsList>& /* vars */,
                          const Variables<DtVarsTagsList>& /* dt_vars */,
                          const tnsr::i<DataVector, VolumeDim, Frame::Inertial>&
                          /* unit_normal_one_form */) noexcept {
    // Memory allocated for return type
    ReturnType& bc_dt_v_plus = get<::Tags::TempScalar<14, DataVector>>(buffer);
    std::fill(bc_dt_v_plus.begin(), bc_dt_v_plus.end(), 0.);
    // Switch on prescribed boundary condition method
    switch (Method) {
      case VPlusBcMethod::Freezing:
        return bc_dt_v_plus;
      case VPlusBcMethod::Unknown:
      default:
        ASSERT(false, "Requested BC method fo VPlus not implemented!");
    }
    // dummy return to suppress compiler warning
    return ReturnType{};
  }

 private:
};

// \brief This struct sets boundary condition on dt<VMinus>
template <typename ReturnType, size_t VolumeDim>
struct set_dt_v_minus {
  template <typename TagsList, typename VarsTagsList, typename DtVarsTagsList>
  static ReturnType apply(const VMinusBcMethod Method,
                          TempBuffer<TagsList>& buffer,
                          const Variables<VarsTagsList>& vars,
                          const Variables<DtVarsTagsList>& /* dt_vars */,
                          const tnsr::i<DataVector, VolumeDim, Frame::Inertial>&
                          /* unit_normal_one_form */) noexcept {
    const auto& bc_dt_v_psi = get<::Tags::TempScalar<6, DataVector>>(buffer);
    const typename Tags::ConstraintGamma2::type& constraint_gamma2 =
        get<::Tags::TempScalar<10, DataVector>>(buffer);

    // testing bc
    const auto char_projected_rhs_dt_v_minus =
        get<::Tags::TempScalar<9, DataVector>>(buffer);
    const tnsr::I<DataVector, VolumeDim,
                  Frame::Inertial>& unit_interface_normal_vector =
        get<::Tags::TempI<0, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    const auto pi = get<Pi>(vars);
    const auto phi = get<Phi<VolumeDim>>(vars);
    const auto psi = get<Psi>(vars);

    // Memory allocated for return type
    ReturnType& bc_dt_v_minus = get<::Tags::TempScalar<15, DataVector>>(buffer);
    std::fill(bc_dt_v_minus.begin(), bc_dt_v_minus.end(), 0.);
    // Switch on prescribed boundary condition method
    switch (Method) {
      case VMinusBcMethod::Freezing:
        return bc_dt_v_minus;
      case VMinusBcMethod::ConstraintPreservingBjorhus:
        return apply_bjorhus_constraint_preserving(
            make_not_null(&bc_dt_v_minus), constraint_gamma2, bc_dt_v_psi);
      case VMinusBcMethod::DampingVMinusBjorhus:
        return apply_bjorhus_damping(
            make_not_null(&bc_dt_v_minus), constraint_gamma2,
            char_projected_rhs_dt_v_minus, unit_interface_normal_vector, pi,
            phi, psi);
      case VMinusBcMethod::Unknown:
      default:
        ASSERT(false, "Requested BC method fo VMinus not implemented!");
    }
    // dummy return to suppress compiler warning
    return ReturnType{};
  }

 private:
  static ReturnType apply_bjorhus_constraint_preserving(
      const gsl::not_null<ReturnType*> bc_dt_v_minus,
      const Scalar<DataVector>& constraint_gamma2,
      const Scalar<DataVector>& bc_dt_v_psi) noexcept {
    ASSERT(get_size(get(*bc_dt_v_minus)) == get_size(get(constraint_gamma2)),
           "Size of input variables and temporary memory do not match.");
    // Note that `bc_dt_v_psi` is the final value of d_t U_\psi that has been
    // set after considering the sign of char speeds of the char field v_\psi
    get(*bc_dt_v_minus) = -get(constraint_gamma2) * get(bc_dt_v_psi);

    return *bc_dt_v_minus;
  }

  static ReturnType apply_bjorhus_damping(
      const gsl::not_null<ReturnType*> bc_dt_v_minus,
      const Scalar<DataVector>& constraint_gamma2,
      const Scalar<DataVector>& /* char_projected_rhs_dt_v_minus */,
      const tnsr::I<DataVector, VolumeDim, Frame::Inertial>
          unit_interface_normal_vector,
      const Scalar<DataVector>& pi,
      const tnsr::i<DataVector, VolumeDim, Frame::Inertial> phi,
      const Scalar<DataVector>& psi) noexcept {
    ASSERT(get_size(get(*bc_dt_v_minus)) == get_size(get(constraint_gamma2)),
           "Size of input variables and temporary memory do not match.");

    // Trying to set dtU
    // = dtU_0 - constant x U
    // = dtU_0 - constant x (Pi - n^k \Phi_k - \gamma2 \psi)
    const double gamma_v_minus = 0.01;
    const auto n_dot_phi = dot_product(unit_interface_normal_vector, phi);
    get(*bc_dt_v_minus) =  // get(char_projected_rhs_dt_v_minus)
        -gamma_v_minus *
        (get(pi) - get(n_dot_phi) - get(constraint_gamma2) * get(psi));

    return *bc_dt_v_minus;
  }
};

}  // namespace BoundaryConditions_detail
}  // namespace Actions
}  // namespace ScalarWave
