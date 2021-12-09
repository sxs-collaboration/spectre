// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <functional>
#include <initializer_list>
#include <optional>
#include <pup.h>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/ObservationBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/FloatingPointType.hpp"
#include "DataStructures/Tensor/TensorData.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/GetSectionObservationKey.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/Tags.hpp"
#include "IO/Observer/VolumeActions.hpp"
#include "NumericalAlgorithms/Interpolation/RegularGridInterpolant.hpp"
#include "Options/Auto.hpp"
#include "Options/Options.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/PupStlCpp17.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

/// \cond
template <size_t Dim>
class Mesh;
namespace Frame {
struct Inertial;
}  // namespace Frame
namespace Tags {
struct Time;
}  // namespace Tags
/// \endcond

namespace dg {
namespace Events {
/// \cond
template <size_t VolumeDim, typename ObservationValueTag, typename Tensors,
          typename NonTensorComputeTagsList = tmpl::list<>,
          typename AnalyticSolutionTensors = tmpl::list<>,
          typename ArraySectionIdTag = void,
          typename NonSolutionTensors =
              tmpl::list_difference<Tensors, AnalyticSolutionTensors>>
class ObserveFields;
/// \endcond

/*!
 * \ingroup DiscontinuousGalerkinGroup
 * \brief %Observe volume tensor fields.
 *
 * A class that writes volume quantities to an h5 file during the simulation.
 * The observed quantitites are:
 * - `InertialCoordinates`
 * - Tensors listed in `Tensors` template parameter
 * - `Error(*)` = errors in `AnalyticSolutionTensors` =
 *   \f$\text{value} - \text{analytic solution}\f$
 *
 * The user may specify an `interpolation_mesh` to which the
 * data is interpolated.
 *
 * \note The `NonTensorComputeTags` are intended to be used for `Variables`
 * compute tags like `Tags::DerivCompute`
 *
 * \par Array sections
 * This event supports sections (see `Parallel::Section`). Set the
 * `ArraySectionIdTag` template parameter to split up observations into subsets
 * of elements. The `observers::Tags::ObservationKey<ArraySectionIdTag>` must be
 * available in the DataBox. It identifies the section and is used as a suffix
 * for the path in the output file.
 */
template <size_t VolumeDim, typename ObservationValueTag, typename... Tensors,
          typename... NonTensorComputeTags, typename... AnalyticSolutionTensors,
          typename ArraySectionIdTag, typename... NonSolutionTensors>
class ObserveFields<VolumeDim, ObservationValueTag, tmpl::list<Tensors...>,
                    tmpl::list<NonTensorComputeTags...>,
                    tmpl::list<AnalyticSolutionTensors...>, ArraySectionIdTag,
                    tmpl::list<NonSolutionTensors...>> : public Event {
 private:
  static_assert(
      std::is_same_v<
          tmpl::list_difference<tmpl::list<AnalyticSolutionTensors...>,
                                tmpl::list<Tensors...>>,
          tmpl::list<>>,
      "All AnalyticSolutionTensors must be listed in Tensors.");
  using coordinates_tag = domain::Tags::Coordinates<VolumeDim, Frame::Inertial>;

 public:
  /// The name of the subfile inside the HDF5 file
  struct SubfileName {
    using type = std::string;
    static constexpr Options::String help = {
        "The name of the subfile inside the HDF5 file without an extension and "
        "without a preceding '/'."};
  };

  /// \cond
  explicit ObserveFields(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ObserveFields);  // NOLINT
  /// \endcond

  struct VariablesToObserve {
    static constexpr Options::String help = "Subset of variables to observe";
    using type = std::vector<std::string>;
    static size_t lower_bound_on_size() { return 1; }
  };

  struct InterpolateToMesh {
    using type = Options::Auto<Mesh<VolumeDim>, Options::AutoLabel::None>;
    static constexpr Options::String help =
        "An optional mesh to which the variables are interpolated. This mesh "
        "specifies any number of collocation points, basis, and quadrature on "
        "which the observed quantities are evaluated. If no mesh is given, the "
        "results will be evaluated on the mesh the simulation runs on. The "
        "user may add several ObserveField Events e.g. with and without an "
        "interpolating mesh to output the data both on the original mesh and "
        "on a new mesh.";
  };

  /// The floating point type/precision with which to write the data to disk.
  ///
  /// Must be specified once for all data or individually for each variable
  /// being observed.
  struct FloatingPointTypes {
    static constexpr Options::String help =
        "The floating point type/precision with which to write the data to "
        "disk.\n\n"
        "Must be specified once for all data or individually  for each "
        "variable being observed.";
    using type = std::vector<FloatingPointType>;
    static size_t upper_bound_on_size() {
      return sizeof...(Tensors) + sizeof...(AnalyticSolutionTensors) +
             sizeof...(NonSolutionTensors);
    }
    static size_t lower_bound_on_size() { return 1; }
  };

  /// The floating point type/precision with which to write the coordinates to
  /// disk.
  struct CoordinatesFloatingPointType {
    static constexpr Options::String help =
        "The floating point type/precision with which to write the coordinates "
        "to disk.";
    using type = FloatingPointType;
  };

  using options =
      tmpl::list<SubfileName, CoordinatesFloatingPointType, FloatingPointTypes,
                 VariablesToObserve, InterpolateToMesh>;

  static constexpr Options::String help =
      "Observe volume tensor fields.\n"
      "\n"
      "Writes volume quantities:\n"
      " * InertialCoordinates\n"
      " * Tensors listed in Tensors template parameter\n"
      " * Error(*) = errors in AnalyticSolutionTensors\n"
      "            = value - analytic solution\n";

  ObserveFields() = default;

  ObserveFields(const std::string& subfile_name,
                FloatingPointType coordinates_floating_point_type,
                const std::vector<FloatingPointType>& floating_point_types,
                const std::vector<std::string>& variables_to_observe,
                std::optional<Mesh<VolumeDim>> interpolation_mesh = {},
                const Options::Context& context = {});

  using compute_tags_for_observation_box =
      tmpl::list<Tensors..., NonTensorComputeTags...>;

  using argument_tags =
      tmpl::list<::Tags::ObservationBox, ObservationValueTag,
                 domain::Tags::Mesh<VolumeDim>, coordinates_tag,
                 AnalyticSolutionTensors..., NonSolutionTensors...>;

  template <typename DataBoxType, typename ComputeTagsList,
            typename Metavariables, typename ParallelComponent>
  void operator()(
      const ObservationBox<DataBoxType, ComputeTagsList>& box,
      const typename ObservationValueTag::type& observation_value,
      const Mesh<VolumeDim>& mesh,
      const tnsr::I<DataVector, VolumeDim, Frame::Inertial>&
          inertial_coordinates,
      const typename AnalyticSolutionTensors::
          type&... analytic_solution_tensors,
      const typename NonSolutionTensors::type&... non_solution_tensors,
      Parallel::GlobalCache<Metavariables>& cache,
      const ElementId<VolumeDim>& array_index,
      const ParallelComponent* const component) const {
    // Skip observation on elements that are not part of a section
    const std::optional<std::string> section_observation_key =
        observers::get_section_observation_key<ArraySectionIdTag>(box);
    if (not section_observation_key.has_value()) {
      return;
    }
    // Get analytic solutions
    auto&& optional_analytic_solutions = [&box]() -> decltype(auto) {
      if constexpr (sizeof...(AnalyticSolutionTensors) > 0) {
        return get<::Tags::AnalyticSolutionsBase>(box);
      } else {
        (void)box;
        return std::nullopt;
      }
    }();
    call_operator_impl(
        subfile_path_ + *section_observation_key, variables_to_observe_,
        interpolation_mesh_, observation_value, mesh, inertial_coordinates,
        analytic_solution_tensors..., non_solution_tensors...,
        optional_analytic_solutions, cache, array_index, component);
  }

  // We factor out the work into a static member function so it can  be shared
  // with other field observing events, like the one that deals with DG-subcell
  // where there are two grids. This is to avoid copy-pasting all of the code.
  template <typename OptionalAnalyticSolutions, typename Metavariables,
            typename ParallelComponent>
  static void call_operator_impl(
      const std::string& subfile_path,
      const std::unordered_map<std::string, FloatingPointType>&
          variables_to_observe,
      const std::optional<Mesh<VolumeDim>>& interpolation_mesh,
      const typename ObservationValueTag::type& observation_value,
      const Mesh<VolumeDim>& mesh,
      const tnsr::I<DataVector, VolumeDim, Frame::Inertial>&
          inertial_coordinates,
      const typename AnalyticSolutionTensors::
          type&... analytic_solution_tensors,
      const typename NonSolutionTensors::type&... non_solution_tensors,
      const OptionalAnalyticSolutions& optional_analytic_solutions,
      Parallel::GlobalCache<Metavariables>& cache,
      const ElementId<VolumeDim>& array_index,
      const ParallelComponent* const /*meta*/) {
    const auto analytic_solutions = [&optional_analytic_solutions]() {
      if constexpr (tt::is_a_v<std::optional, OptionalAnalyticSolutions>) {
        return optional_analytic_solutions.has_value()
                   ? std::make_optional(std::cref(*optional_analytic_solutions))
                   : std::nullopt;
      } else {
        return std::make_optional(std::cref(optional_analytic_solutions));
      }
    }();

    const std::string element_name =
        MakeString{} << ElementId<VolumeDim>(array_index) << '/';

    // if no interpolation_mesh is provided, the interpolation is essentially
    // ignored by the RegularGridInterpolant except for a single copy.
    const intrp::RegularGrid interpolant(mesh,
                                         interpolation_mesh.value_or(mesh));

    // Remove tensor types, only storing individual components.
    std::vector<TensorComponent> components;
    // This is larger than we need if we are only observing some
    // tensors, but that's not a big deal and calculating the correct
    // size is nontrivial.
    components.reserve(alg::accumulate(
        std::initializer_list<size_t>{
            inertial_coordinates.size(),
            (analytic_solutions.has_value() ? 2 : 1) *
                AnalyticSolutionTensors::type::size()...,
            NonSolutionTensors::type::size()...},
        0_st));

    const auto record_tensor_components = [&components, &element_name,
                                           &interpolant, &variables_to_observe](
                                              const auto tensor_tag_v,
                                              const auto& tensor) {
      using tensor_tag = tmpl::type_from<decltype(tensor_tag_v)>;
      if (variables_to_observe.count(db::tag_name<tensor_tag>()) == 1) {
        const auto floating_point_type =
            variables_to_observe.at(db::tag_name<tensor_tag>());
        for (size_t i = 0; i < tensor.size(); ++i) {
          const auto tensor_component = interpolant.interpolate(tensor[i]);
          if (floating_point_type == FloatingPointType::Float) {
            components.emplace_back(element_name + db::tag_name<tensor_tag>() +
                                        tensor.component_suffix(i),
                                    std::vector<float>{tensor_component.begin(),
                                                       tensor_component.end()});
          } else {
            components.emplace_back(element_name + db::tag_name<tensor_tag>() +
                                        tensor.component_suffix(i),
                                    tensor_component);
          }
        }
      }
    };
    record_tensor_components(tmpl::type_<coordinates_tag>{},
                             inertial_coordinates);
    EXPAND_PACK_LEFT_TO_RIGHT(record_tensor_components(
        tmpl::type_<AnalyticSolutionTensors>{}, analytic_solution_tensors));
    EXPAND_PACK_LEFT_TO_RIGHT(record_tensor_components(
        tmpl::type_<NonSolutionTensors>{}, non_solution_tensors));

    if (analytic_solutions.has_value()) {
      const auto record_errors =
          [&analytic_solutions, &components, &element_name, &interpolant,
           &variables_to_observe](const auto tensor_tag_v, const auto& tensor) {
            using tensor_tag = tmpl::type_from<decltype(tensor_tag_v)>;
            if (variables_to_observe.count(db::tag_name<tensor_tag>()) == 1) {
              const auto floating_point_type =
                  variables_to_observe.at(db::tag_name<tensor_tag>());
              for (size_t i = 0; i < tensor.size(); ++i) {
                DataVector error = interpolant.interpolate(
                    DataVector(tensor[i] - get<::Tags::Analytic<tensor_tag>>(
                                               analytic_solutions->get())[i]));
                if (floating_point_type == FloatingPointType::Float) {
                  components.emplace_back(
                      element_name + "Error(" + db::tag_name<tensor_tag>() +
                          ")" + tensor.component_suffix(i),
                      std::vector<float>{error.begin(), error.end()});
                } else {
                  components.emplace_back(element_name + "Error(" +
                                              db::tag_name<tensor_tag>() + ")" +
                                              tensor.component_suffix(i),
                                          std::move(error));
                }
              }
            }
          };
      EXPAND_PACK_LEFT_TO_RIGHT(record_errors(
          tmpl::type_<AnalyticSolutionTensors>{}, analytic_solution_tensors));

      (void)(record_errors);  // Silence GCC warning about unused variable
    }

    // Send data to volume observer
    auto& local_observer =
        *Parallel::get_parallel_component<observers::Observer<Metavariables>>(
             cache)
             .ckLocalBranch();
    Parallel::simple_action<observers::Actions::ContributeVolumeData>(
        local_observer,
        observers::ObservationId(observation_value, subfile_path + ".vol"),
        subfile_path,
        observers::ArrayComponentId(
            std::add_pointer_t<ParallelComponent>{nullptr},
            Parallel::ArrayIndex<ElementId<VolumeDim>>(array_index)),
        std::move(components), interpolation_mesh.value_or(mesh).extents(),
        interpolation_mesh.value_or(mesh).basis(),
        interpolation_mesh.value_or(mesh).quadrature());
  }

  using observation_registration_tags = tmpl::list<::Tags::DataBox>;

  template <typename DbTagsList, typename Metavariables,
            typename ParallelComponent>
  std::optional<
      std::pair<observers::TypeOfObservation, observers::ObservationKey>>
  get_observation_type_and_key_for_registration(
      const db::DataBox<DbTagsList>& box,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<VolumeDim>& /*array_index*/,
      const ParallelComponent* const /*meta*/) const {
    const std::optional<std::string> section_observation_key =
        observers::get_section_observation_key<ArraySectionIdTag>(box);
    if (not section_observation_key.has_value()) {
      return std::nullopt;
    }
    return {{observers::TypeOfObservation::Volume,
             observers::ObservationKey(
                 subfile_path_ + section_observation_key.value() + ".vol")}};
  }

  using is_ready_argument_tags = tmpl::list<>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  bool is_ready(Parallel::GlobalCache<Metavariables>& /*cache*/,
                const ArrayIndex& /*array_index*/,
                const Component* const /*meta*/) const {
    return true;
  }

  bool needs_evolved_variables() const override { return true; }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override {
    Event::pup(p);
    p | subfile_path_;
    p | variables_to_observe_;
    p | interpolation_mesh_;
  }

 private:
  std::string subfile_path_;
  std::unordered_map<std::string, FloatingPointType> variables_to_observe_{};
  std::optional<Mesh<VolumeDim>> interpolation_mesh_{};
};

template <size_t VolumeDim, typename ObservationValueTag, typename... Tensors,
          typename... NonTensorComputeTags, typename... AnalyticSolutionTensors,
          typename ArraySectionIdTag, typename... NonSolutionTensors>
ObserveFields<VolumeDim, ObservationValueTag, tmpl::list<Tensors...>,
              tmpl::list<NonTensorComputeTags...>,
              tmpl::list<AnalyticSolutionTensors...>, ArraySectionIdTag,
              tmpl::list<NonSolutionTensors...>>::
    ObserveFields(const std::string& subfile_name,
                  const FloatingPointType coordinates_floating_point_type,
                  const std::vector<FloatingPointType>& floating_point_types,
                  const std::vector<std::string>& variables_to_observe,
                  std::optional<Mesh<VolumeDim>> interpolation_mesh,
                  const Options::Context& context)
    : subfile_path_("/" + subfile_name),
      variables_to_observe_([&context, &floating_point_types,
                             &variables_to_observe]() {
        if (floating_point_types.size() != 1 and
            floating_point_types.size() != variables_to_observe.size()) {
          PARSE_ERROR(context, "The number of floating point types specified ("
                                   << floating_point_types.size()
                                   << ") must be 1 or the number of variables "
                                      "specified for observing ("
                                   << variables_to_observe.size() << ")");
        }
        std::unordered_map<std::string, FloatingPointType> result{};
        for (size_t i = 0; i < variables_to_observe.size(); ++i) {
          result[variables_to_observe[i]] = floating_point_types.size() == 1
                                                ? floating_point_types[0]
                                                : floating_point_types[i];
          ASSERT(
              result.at(variables_to_observe[i]) == FloatingPointType::Float or
                  result.at(variables_to_observe[i]) ==
                      FloatingPointType::Double,
              "Floating point type for variable '"
                  << variables_to_observe[i]
                  << "' must be either Float or Double.");
        }
        return result;
      }()),
      interpolation_mesh_(interpolation_mesh) {
  using ::operator<<;
  const std::unordered_set<std::string> valid_tensors{
      db::tag_name<Tensors>()...};
  for (const auto& [name, floating_point_type] : variables_to_observe_) {
    (void)floating_point_type;
    if (valid_tensors.count(name) != 1) {
      PARSE_ERROR(
          context,
          name << " is not an available variable.  Available variables:\n"
               << (std::vector<std::string>{db::tag_name<Tensors>()...}));
    }
    if (alg::count(variables_to_observe, name) != 1) {
      PARSE_ERROR(context, name << " specified multiple times");
    }
  }
  variables_to_observe_[coordinates_tag::name()] =
      coordinates_floating_point_type;
}

/// \cond
template <size_t VolumeDim, typename ObservationValueTag, typename... Tensors,
          typename... NonTensorComputeTags, typename... AnalyticSolutionTensors,
          typename ArraySectionIdTag, typename... NonSolutionTensors>
PUP::able::PUP_ID
    ObserveFields<VolumeDim, ObservationValueTag, tmpl::list<Tensors...>,
                  tmpl::list<NonTensorComputeTags...>,
                  tmpl::list<AnalyticSolutionTensors...>, ArraySectionIdTag,
                  tmpl::list<NonSolutionTensors...>>::my_PUP_ID = 0;  // NOLINT
/// \endcond
}  // namespace Events
}  // namespace dg
