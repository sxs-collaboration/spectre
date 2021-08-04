// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/Limiters/Weno.hpp"

#include <array>
#include <boost/functional/hash.hpp>  // IWYU pragma: keep
#include <cstddef>
#include <optional>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/SizeOfElement.hpp"
#include "Domain/Structure/Direction.hpp"  // IWYU pragma: keep
#include "Domain/Structure/Element.hpp"    // IWYU pragma: keep
#include "Domain/Structure/ElementId.hpp"  // IWYU pragma: keep
#include "Domain/Tags.hpp"                 // IWYU pragma: keep
#include "Evolution/DiscontinuousGalerkin/Limiters/HwenoImpl.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/MinmodHelpers.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/MinmodTci.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/SimpleWenoImpl.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/Weno.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/WenoType.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Limiters/CharacteristicHelpers.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Limiters/Flattener.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Limiters/KxrcfTci.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Limiters/VariablesToLimit.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/RegularGridInterpolant.hpp"
#include "NumericalAlgorithms/LinearOperators/MeanValue.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <size_t ThermodynamicDim>
bool characteristic_simple_weno_impl(
    const gsl::not_null<Scalar<DataVector>*> tilde_d,
    const gsl::not_null<Scalar<DataVector>*> tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3>*> tilde_s,
    const gsl::not_null<tnsr::I<DataVector, 3>*> tilde_b,
    const gsl::not_null<Scalar<DataVector>*> tilde_phi,
    const double tvb_constant, const double neighbor_linear_weight,
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift,
    const tnsr::ii<DataVector, 3>& spatial_metric, const Mesh<3>& mesh,
    const Element<3>& element, const std::array<double, 3>& element_size,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state,
    const std::unordered_map<
        std::pair<Direction<3>, ElementId<3>>,
        typename grmhd::ValenciaDivClean::Limiters::Weno::PackagedData,
        boost::hash<std::pair<Direction<3>, ElementId<3>>>>&
        neighbor_data) noexcept {
  // Storage for transforming neighbor_data into char variables
  using CharacteristicVarsWeno =
      Limiters::Weno<3,
                     tmpl::list<grmhd::ValenciaDivClean::Tags::VDivCleanMinus,
                                grmhd::ValenciaDivClean::Tags::VMinus,
                                grmhd::ValenciaDivClean::Tags::VMomentum,
                                grmhd::ValenciaDivClean::Tags::VPlus,
                                grmhd::ValenciaDivClean::Tags::VDivCleanPlus>>;
  std::unordered_map<std::pair<Direction<3>, ElementId<3>>,
                     typename CharacteristicVarsWeno::PackagedData,
                     boost::hash<std::pair<Direction<3>, ElementId<3>>>>
      neighbor_char_data{};
  for (const auto& [key, data] : neighbor_data) {
    neighbor_char_data[key].volume_data.initialize(
        mesh.number_of_grid_points());
    neighbor_char_data[key].mesh = data.mesh;
    neighbor_char_data[key].element_size = data.element_size;
  }

  // Buffers for TCI
  Limiters::Minmod_detail::BufferWrapper<3> tci_buffer(mesh);
  const auto effective_neighbor_sizes =
      Limiters::Minmod_detail::compute_effective_neighbor_sizes(element,
                                                                neighbor_data);

  // Buffers for SimpleWeno extrapolated poly
  std::unordered_map<std::pair<Direction<3>, ElementId<3>>,
                     intrp::RegularGrid<3>,
                     boost::hash<std::pair<Direction<3>, ElementId<3>>>>
      interpolator_buffer{};
  std::unordered_map<std::pair<Direction<3>, ElementId<3>>, DataVector,
                     boost::hash<std::pair<Direction<3>, ElementId<3>>>>
      modified_neighbor_solution_buffer{};

  // Outer lambda: wraps applying SimpleWeno to the ValenciaDivClean
  // characteristics for one particular choice of characteristic decomposition
  const auto simple_weno_convert_neighbor_data_then_limit =
      [&tci_buffer, &interpolator_buffer, &modified_neighbor_solution_buffer,
       &tvb_constant, &neighbor_linear_weight, &mesh, &element, &element_size,
       &neighbor_data, &neighbor_char_data, &effective_neighbor_sizes](
          const gsl::not_null<Scalar<DataVector>*> char_v_div_clean_minus,
          const gsl::not_null<Scalar<DataVector>*> char_v_minus,
          const gsl::not_null<tnsr::I<DataVector, 5>*> char_v_momentum,
          const gsl::not_null<Scalar<DataVector>*> char_v_plus,
          const gsl::not_null<Scalar<DataVector>*> char_v_div_clean_plus,
          const Matrix& left_eigenvectors) noexcept -> bool {
    // Convert neighbor data to characteristics
    for (const auto& [key, data] : neighbor_data) {
      grmhd::ValenciaDivClean::Limiters::characteristic_fields(
          make_not_null(&(neighbor_char_data[key].volume_data)),
          data.volume_data, left_eigenvectors);
      grmhd::ValenciaDivClean::Limiters::characteristic_fields(
          make_not_null(&(neighbor_char_data[key].means)), data.means,
          left_eigenvectors);
    }

    bool some_component_was_limited_with_this_normal = false;

    // Inner lambda: apply SimpleWeno to one particular tensor
    const auto wrap_minmod_tci_and_simple_weno =
        [&some_component_was_limited_with_this_normal, &tci_buffer,
         &interpolator_buffer, &modified_neighbor_solution_buffer,
         &tvb_constant, &neighbor_linear_weight, &mesh, &element, &element_size,
         &neighbor_char_data,
         &effective_neighbor_sizes](auto tag, const auto tensor) noexcept {
          for (size_t tensor_storage_index = 0;
               tensor_storage_index < tensor->size(); ++tensor_storage_index) {
            // Check TCI
            const auto effective_neighbor_means =
                Limiters::Minmod_detail::compute_effective_neighbor_means<
                    decltype(tag)>(tensor_storage_index, element,
                                   neighbor_char_data);
            const bool component_needs_limiting =
                Limiters::Tci::tvb_minmod_indicator(
                    make_not_null(&tci_buffer), tvb_constant,
                    (*tensor)[tensor_storage_index], mesh, element,
                    element_size, effective_neighbor_means,
                    effective_neighbor_sizes);

            if (component_needs_limiting) {
              if (modified_neighbor_solution_buffer.empty()) {
                // Allocate the neighbor solution buffers only if the limiter is
                // triggered. This reduces allocation when no limiting occurs.
                for (const auto& [neighbor, data] : neighbor_char_data) {
                  (void)data;
                  modified_neighbor_solution_buffer.insert(std::make_pair(
                      neighbor, DataVector(mesh.number_of_grid_points())));
                }
              }
              Limiters::Weno_detail::simple_weno_impl<decltype(tag)>(
                  make_not_null(&interpolator_buffer),
                  make_not_null(&modified_neighbor_solution_buffer), tensor,
                  neighbor_linear_weight, tensor_storage_index, mesh, element,
                  neighbor_char_data);
              some_component_was_limited_with_this_normal = true;
            }
          }
        };
    wrap_minmod_tci_and_simple_weno(
        grmhd::ValenciaDivClean::Tags::VDivCleanMinus{},
        char_v_div_clean_minus);
    wrap_minmod_tci_and_simple_weno(grmhd::ValenciaDivClean::Tags::VMinus{},
                                    char_v_minus);
    wrap_minmod_tci_and_simple_weno(grmhd::ValenciaDivClean::Tags::VMomentum{},
                                    char_v_momentum);
    wrap_minmod_tci_and_simple_weno(grmhd::ValenciaDivClean::Tags::VPlus{},
                                    char_v_plus);
    wrap_minmod_tci_and_simple_weno(
        grmhd::ValenciaDivClean::Tags::VDivCleanPlus{}, char_v_div_clean_plus);
    return some_component_was_limited_with_this_normal;
  };

  return grmhd::ValenciaDivClean::Limiters::
      apply_limiter_to_characteristic_fields_in_all_directions(
          tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi, lapse, shift,
          spatial_metric, mesh, equation_of_state,
          simple_weno_convert_neighbor_data_then_limit);
}

template <size_t ThermodynamicDim>
bool characteristic_hweno_impl(
    const gsl::not_null<Scalar<DataVector>*> tilde_d,
    const gsl::not_null<Scalar<DataVector>*> tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3>*> tilde_s,
    const gsl::not_null<tnsr::I<DataVector, 3>*> tilde_b,
    const gsl::not_null<Scalar<DataVector>*> tilde_phi,
    const double kxrcf_constant, const double neighbor_linear_weight,
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift,
    const tnsr::ii<DataVector, 3>& spatial_metric, const Mesh<3>& mesh,
    const Element<3>& element, const std::array<double, 3>& element_size,
    const Scalar<DataVector>& det_logical_to_inertial_jacobian,
    const typename evolution::dg::Tags::NormalCovectorAndMagnitude<3>::type&
        normals_and_magnitudes,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state,
    const std::unordered_map<
        std::pair<Direction<3>, ElementId<3>>,
        typename grmhd::ValenciaDivClean::Limiters::Weno::PackagedData,
        boost::hash<std::pair<Direction<3>, ElementId<3>>>>&
        neighbor_data) noexcept {
  // Hweno checks TCI before limiting any tensors
  const bool cell_is_troubled =
      grmhd::ValenciaDivClean::Limiters::Tci::kxrcf_indicator(
          kxrcf_constant, *tilde_d, *tilde_tau, *tilde_s, mesh, element,
          element_size, det_logical_to_inertial_jacobian,
          normals_and_magnitudes, neighbor_data);
  if (not cell_is_troubled) {
    // No limiting is needed
    return false;
  }

  // Storage for transforming neighbor_data into char variables
  using CharacteristicVarsWeno =
      Limiters::Weno<3,
                     tmpl::list<grmhd::ValenciaDivClean::Tags::VDivCleanMinus,
                                grmhd::ValenciaDivClean::Tags::VMinus,
                                grmhd::ValenciaDivClean::Tags::VMomentum,
                                grmhd::ValenciaDivClean::Tags::VPlus,
                                grmhd::ValenciaDivClean::Tags::VDivCleanPlus>>;
  std::unordered_map<std::pair<Direction<3>, ElementId<3>>,
                     typename CharacteristicVarsWeno::PackagedData,
                     boost::hash<std::pair<Direction<3>, ElementId<3>>>>
      neighbor_char_data{};
  for (const auto& [key, data] : neighbor_data) {
    neighbor_char_data[key].volume_data.initialize(
        mesh.number_of_grid_points());
    neighbor_char_data[key].mesh = data.mesh;
    neighbor_char_data[key].element_size = data.element_size;
  }

  // Buffers for Hweno extrapolated poly
  std::unordered_map<std::pair<Direction<3>, ElementId<3>>, DataVector,
                     boost::hash<std::pair<Direction<3>, ElementId<3>>>>
      modified_neighbor_solution_buffer{};
  for (const auto& [neighbor, data] : neighbor_char_data) {
    (void)data;
    modified_neighbor_solution_buffer.insert(
        std::make_pair(neighbor, DataVector(mesh.number_of_grid_points())));
  }

  // Lambda wraps applying Hweno to the ValenciaDivClean characteristics for one
  // particular choice of characteristic decomposition
  const auto hweno_convert_neighbor_data_then_limit =
      [&modified_neighbor_solution_buffer, &neighbor_linear_weight, &mesh,
       &element, &neighbor_data, &neighbor_char_data](
          const gsl::not_null<Scalar<DataVector>*> char_v_div_clean_minus,
          const gsl::not_null<Scalar<DataVector>*> char_v_minus,
          const gsl::not_null<tnsr::I<DataVector, 5>*> char_v_momentum,
          const gsl::not_null<Scalar<DataVector>*> char_v_plus,
          const gsl::not_null<Scalar<DataVector>*> char_v_div_clean_plus,
          const Matrix& left_eigenvectors) noexcept -> bool {
    // Convert neighbor data to characteristics
    for (const auto& [key, data] : neighbor_data) {
      grmhd::ValenciaDivClean::Limiters::characteristic_fields(
          make_not_null(&(neighbor_char_data[key].volume_data)),
          data.volume_data, left_eigenvectors);
      grmhd::ValenciaDivClean::Limiters::characteristic_fields(
          make_not_null(&(neighbor_char_data[key].means)), data.means,
          left_eigenvectors);
    }

    ::Limiters::Weno_detail::hweno_impl<
        grmhd::ValenciaDivClean::Tags::VDivCleanMinus>(
        make_not_null(&modified_neighbor_solution_buffer),
        char_v_div_clean_minus, neighbor_linear_weight, mesh, element,
        neighbor_char_data);
    ::Limiters::Weno_detail::hweno_impl<grmhd::ValenciaDivClean::Tags::VMinus>(
        make_not_null(&modified_neighbor_solution_buffer), char_v_minus,
        neighbor_linear_weight, mesh, element, neighbor_char_data);
    ::Limiters::Weno_detail::hweno_impl<
        grmhd::ValenciaDivClean::Tags::VMomentum>(
        make_not_null(&modified_neighbor_solution_buffer), char_v_momentum,
        neighbor_linear_weight, mesh, element, neighbor_char_data);
    ::Limiters::Weno_detail::hweno_impl<grmhd::ValenciaDivClean::Tags::VPlus>(
        make_not_null(&modified_neighbor_solution_buffer), char_v_plus,
        neighbor_linear_weight, mesh, element, neighbor_char_data);
    ::Limiters::Weno_detail::hweno_impl<
        grmhd::ValenciaDivClean::Tags::VDivCleanPlus>(
        make_not_null(&modified_neighbor_solution_buffer),
        char_v_div_clean_plus, neighbor_linear_weight, mesh, element,
        neighbor_char_data);
    return true;  // all components were limited
  };

  grmhd::ValenciaDivClean::Limiters::
      apply_limiter_to_characteristic_fields_in_all_directions(
          tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi, lapse, shift,
          spatial_metric, mesh, equation_of_state,
          hweno_convert_neighbor_data_then_limit);
  return true;  // all components were limited
}

bool conservative_hweno_impl(
    const gsl::not_null<Scalar<DataVector>*> tilde_d,
    const gsl::not_null<Scalar<DataVector>*> tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3>*> tilde_s,
    const gsl::not_null<tnsr::I<DataVector, 3>*> tilde_b,
    const gsl::not_null<Scalar<DataVector>*> tilde_phi,
    const double kxrcf_constant, const double neighbor_linear_weight,
    const Mesh<3>& mesh, const Element<3>& element,
    const std::array<double, 3>& element_size,
    const Scalar<DataVector>& det_logical_to_inertial_jacobian,
    const typename evolution::dg::Tags::NormalCovectorAndMagnitude<3>::type&
        normals_and_magnitudes,
    const std::unordered_map<
        std::pair<Direction<3>, ElementId<3>>,
        typename grmhd::ValenciaDivClean::Limiters::Weno::PackagedData,
        boost::hash<std::pair<Direction<3>, ElementId<3>>>>&
        neighbor_data) noexcept {
  // Hweno checks TCI before limiting any tensors
  const bool cell_is_troubled =
      grmhd::ValenciaDivClean::Limiters::Tci::kxrcf_indicator(
          kxrcf_constant, *tilde_d, *tilde_tau, *tilde_s, mesh, element,
          element_size, det_logical_to_inertial_jacobian,
          normals_and_magnitudes, neighbor_data);
  if (not cell_is_troubled) {
    // No limiting is needed
    return false;
  }

  // Buffers for Hweno extrapolated poly
  std::unordered_map<std::pair<Direction<3>, ElementId<3>>, DataVector,
                     boost::hash<std::pair<Direction<3>, ElementId<3>>>>
      modified_neighbor_solution_buffer{};
  for (const auto& [neighbor, data] : neighbor_data) {
    (void)data;
    modified_neighbor_solution_buffer.insert(
        std::make_pair(neighbor, DataVector(mesh.number_of_grid_points())));
  }

  ::Limiters::Weno_detail::hweno_impl<grmhd::ValenciaDivClean::Tags::TildeD>(
      make_not_null(&modified_neighbor_solution_buffer), tilde_d,
      neighbor_linear_weight, mesh, element, neighbor_data);
  ::Limiters::Weno_detail::hweno_impl<grmhd::ValenciaDivClean::Tags::TildeTau>(
      make_not_null(&modified_neighbor_solution_buffer), tilde_tau,
      neighbor_linear_weight, mesh, element, neighbor_data);
  ::Limiters::Weno_detail::hweno_impl<grmhd::ValenciaDivClean::Tags::TildeS<>>(
      make_not_null(&modified_neighbor_solution_buffer), tilde_s,
      neighbor_linear_weight, mesh, element, neighbor_data);
  ::Limiters::Weno_detail::hweno_impl<grmhd::ValenciaDivClean::Tags::TildeB<>>(
      make_not_null(&modified_neighbor_solution_buffer), tilde_b,
      neighbor_linear_weight, mesh, element, neighbor_data);
  ::Limiters::Weno_detail::hweno_impl<grmhd::ValenciaDivClean::Tags::TildePhi>(
      make_not_null(&modified_neighbor_solution_buffer), tilde_phi,
      neighbor_linear_weight, mesh, element, neighbor_data);
  return true;  // all components were limited
}

}  // namespace

namespace grmhd::ValenciaDivClean::Limiters {

Weno::Weno(
    const ::Limiters::WenoType weno_type,
    const grmhd::ValenciaDivClean::Limiters::VariablesToLimit vars_to_limit,
    const double neighbor_linear_weight,
    const std::optional<double> tvb_constant,
    const std::optional<double> kxrcf_constant, const bool apply_flattener,
    const bool disable_for_debugging, const Options::Context& context)
    : weno_type_(weno_type),
      vars_to_limit_(vars_to_limit),
      neighbor_linear_weight_(neighbor_linear_weight),
      tvb_constant_(tvb_constant),
      kxrcf_constant_(kxrcf_constant),
      apply_flattener_(apply_flattener),
      disable_for_debugging_(disable_for_debugging),
      conservative_vars_weno_(
          weno_type_, neighbor_linear_weight_,
          tvb_constant_.value_or(std::numeric_limits<double>::signaling_NaN()),
          disable_for_debugging_) {
  if (weno_type == ::Limiters::WenoType::Hweno) {
    if (tvb_constant.has_value() or not kxrcf_constant.has_value()) {
      PARSE_ERROR(context,
                  "The Hweno limiter uses the KXRCF TCI. The TvbConstant must "
                  "be set to 'None', and the KxrcfConstant must be set to a "
                  "non-negative value.");
    }
    if (kxrcf_constant.value() < 0.0) {
      PARSE_ERROR(context, "The KXRCF constant must be non-negative, but got: "
                               << kxrcf_constant.value());
    }
  } else {  // SimpleWeno
    if (not tvb_constant.has_value() or kxrcf_constant.has_value()) {
      PARSE_ERROR(context,
                  "The SimpleWeno limiter uses the TVB minmod TCI. The "
                  "TvbConstant must be set to a non-negative value, and the "
                  "KxrcfConstant must be set to 'None'.");
    }
    if (tvb_constant.value() < 0.0) {
      PARSE_ERROR(context, "The TVB constant must be non-negative, but got: "
                               << tvb_constant.value());
    }
  }
}

// NOLINTNEXTLINE(google-runtime-references)
void Weno::pup(PUP::er& p) noexcept {
  p | weno_type_;
  p | vars_to_limit_;
  p | neighbor_linear_weight_;
  p | tvb_constant_;
  p | kxrcf_constant_;
  p | apply_flattener_;
  p | disable_for_debugging_;
}

void Weno::package_data(
    const gsl::not_null<PackagedData*> packaged_data,
    const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_tau,
    const tnsr::i<DataVector, 3>& tilde_s,
    const tnsr::I<DataVector, 3>& tilde_b, const Scalar<DataVector>& tilde_phi,
    const Mesh<3>& mesh, const std::array<double, 3>& element_size,
    const OrientationMap<3>& orientation_map) const noexcept {
  conservative_vars_weno_.package_data(packaged_data, tilde_d, tilde_tau,
                                       tilde_s, tilde_b, tilde_phi, mesh,
                                       element_size, orientation_map);
}

template <size_t ThermodynamicDim>
bool Weno::operator()(
    const gsl::not_null<Scalar<DataVector>*> tilde_d,
    const gsl::not_null<Scalar<DataVector>*> tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3>*> tilde_s,
    const gsl::not_null<tnsr::I<DataVector, 3>*> tilde_b,
    const gsl::not_null<Scalar<DataVector>*> tilde_phi,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift,
    const tnsr::ii<DataVector, 3>& spatial_metric, const Mesh<3>& mesh,
    const Element<3>& element, const std::array<double, 3>& element_size,
    const Scalar<DataVector>& det_inv_logical_to_inertial_jacobian,
    const typename evolution::dg::Tags::NormalCovectorAndMagnitude<3>::type&
        normals_and_magnitudes,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state,
    const std::unordered_map<
        std::pair<Direction<3>, ElementId<3>>, PackagedData,
        boost::hash<std::pair<Direction<3>, ElementId<3>>>>& neighbor_data)
    const noexcept {
  if (UNLIKELY(disable_for_debugging_)) {
    // Do not modify input tensors
    return false;
  }

  // Enforce restrictions on h-refinement, p-refinement
  if (UNLIKELY(alg::any_of(element.neighbors(),
                           [](const auto& direction_neighbors) noexcept {
                             return direction_neighbors.second.size() != 1;
                           }))) {
    ERROR("The Weno limiter does not yet support h-refinement");
    // Removing this limitation will require:
    // - Generalizing the computation of the modified neighbor solutions.
    // - Generalizing the WENO weighted sum for multiple neighbors in each
    //   direction.
  }
  alg::for_each(neighbor_data, [&mesh](const auto& neighbor_and_data) noexcept {
    if (UNLIKELY(neighbor_and_data.second.mesh != mesh)) {
      ERROR("The Weno limiter does not yet support p-refinement");
      // Removing this limitation will require generalizing the
      // computation of the modified neighbor solutions.
    }
  });

  // Checks for the post-timestep, pre-limiter ValenciaDivClean state:
#ifdef SPECTRE_DEBUG
  const double mean_tilde_d = mean_value(get(*tilde_d), mesh);
  ASSERT(mean_tilde_d > 0.0,
         "Positivity was violated on a cell-average level.");
  // TODO: Add other simple positivity checks?
#endif  // SPECTRE_DEBUG

  bool limiter_activated = false;

  // Small possible optimization: only compute this if needed
  const Scalar<DataVector> det_logical_to_inertial_jacobian{
      1. / get(det_inv_logical_to_inertial_jacobian)};

  if (weno_type_ == ::Limiters::WenoType::Hweno) {
    if (vars_to_limit_ == grmhd::ValenciaDivClean::Limiters::VariablesToLimit::
                              NumericalCharacteristic) {
      // impl function handles specialized TCI + char transform
      limiter_activated = characteristic_hweno_impl(
          tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi,
          kxrcf_constant_.value(), neighbor_linear_weight_, lapse, shift,
          spatial_metric, mesh, element, element_size,
          det_logical_to_inertial_jacobian, normals_and_magnitudes,
          equation_of_state, neighbor_data);
    } else {
      // impl function handles specialized TCI
      limiter_activated = conservative_hweno_impl(
          tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi,
          kxrcf_constant_.value(), neighbor_linear_weight_, mesh, element,
          element_size, det_logical_to_inertial_jacobian,
          normals_and_magnitudes, neighbor_data);
    }
  } else if (weno_type_ == ::Limiters::WenoType::SimpleWeno) {
    if (vars_to_limit_ == grmhd::ValenciaDivClean::Limiters::VariablesToLimit::
                              NumericalCharacteristic) {
      // impl function handles char transform
      limiter_activated = characteristic_simple_weno_impl(
          tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi,
          tvb_constant_.value(), neighbor_linear_weight_, lapse, shift,
          spatial_metric, mesh, element, element_size, equation_of_state,
          neighbor_data);
    } else {
      // Fall back to generic SimpleWeno
      limiter_activated = conservative_vars_weno_(
          tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi, mesh, element,
          element_size, neighbor_data);
    }
  }

  if (apply_flattener_) {
    const auto flattener_action = flatten_solution(
        tilde_d, tilde_tau, tilde_s, *tilde_b, sqrt_det_spatial_metric,
        spatial_metric, mesh, det_logical_to_inertial_jacobian);
    if (flattener_action != FlattenerAction::NoOp) {
      limiter_activated = true;
    }
  }

  // Checks for the post-limiter ValenciaDivClean state:
#ifdef SPECTRE_DEBUG
  ASSERT(min(get(*tilde_d)) > 0.0, "Bad density after limiting.");
  // TODO: Add other simple positivity checks?
#endif  // SPECTRE_DEBUG

  return limiter_activated;
}

bool operator==(const Weno& lhs, const Weno& rhs) noexcept {
  // No need to compare the conservative_vars_weno_ member variable because
  // it is constructed from the other member variables.
  return lhs.weno_type_ == rhs.weno_type_ and
         lhs.vars_to_limit_ == rhs.vars_to_limit_ and
         lhs.neighbor_linear_weight_ == rhs.neighbor_linear_weight_ and
         lhs.tvb_constant_ == rhs.tvb_constant_ and
         lhs.kxrcf_constant_ == rhs.kxrcf_constant_ and
         lhs.apply_flattener_ == rhs.apply_flattener_ and
         lhs.disable_for_debugging_ == rhs.disable_for_debugging_;
}

bool operator!=(const Weno& lhs, const Weno& rhs) noexcept {
  return not(lhs == rhs);
}

#define THERMO_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                               \
  template bool Weno::operator()(                                          \
      const gsl::not_null<Scalar<DataVector>*>,                            \
      const gsl::not_null<Scalar<DataVector>*>,                            \
      const gsl::not_null<tnsr::i<DataVector, 3>*>,                        \
      const gsl::not_null<tnsr::I<DataVector, 3>*>,                        \
      const gsl::not_null<Scalar<DataVector>*>, const Scalar<DataVector>&, \
      const Scalar<DataVector>&, const tnsr::I<DataVector, 3>&,            \
      const tnsr::ii<DataVector, 3>&, const Mesh<3>&, const Element<3>&,   \
      const std::array<double, 3>&, const Scalar<DataVector>&,             \
      const typename evolution::dg::Tags::NormalCovectorAndMagnitude<      \
          3>::type&,                                                       \
      const EquationsOfState::EquationOfState<true, THERMO_DIM(data)>&,    \
      const std::unordered_map<                                            \
          std::pair<Direction<3>, ElementId<3>>, PackagedData,             \
          boost::hash<std::pair<Direction<3>, ElementId<3>>>>&)            \
      const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2))

#undef INSTANTIATE
#undef THERMO_DIM

}  // namespace grmhd::ValenciaDivClean::Limiters
