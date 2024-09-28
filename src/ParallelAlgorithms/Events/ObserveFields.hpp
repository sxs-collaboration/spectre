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
#include "DataStructures/DataBox/ValidateSelection.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/FloatingPointType.hpp"
#include "Domain/Structure/BlockGroups.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "IO/H5/TensorData.hpp"
#include "IO/Observer/GetSectionObservationKey.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/Tags.hpp"
#include "IO/Observer/VolumeActions.hpp"
#include "NumericalAlgorithms/Interpolation/RegularGridInterpolant.hpp"
#include "Options/Auto.hpp"
#include "Options/String.hpp"
#include "Parallel/ArrayComponentId.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "ParallelAlgorithms/Events/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/OptionalHelpers.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

/// \cond
template <size_t Dim>
class Mesh;
namespace Frame {
struct Inertial;
}  // namespace Frame
/// \endcond

namespace dg {
namespace Events {
/// \cond
template <size_t VolumeDim, typename Tensors,
          typename NonTensorComputeTagsList = tmpl::list<>,
          typename ArraySectionIdTag = void>
class ObserveFields;
/// \endcond

/*!
 * \ingroup DiscontinuousGalerkinGroup
 * \brief %Observe volume tensor fields.
 *
 * A class that writes volume quantities to an h5 file during the simulation.
 * The observed quantitites are specified in the `VariablesToObserve` option.
 * Any `Tensor` in the `db::DataBox` can be observed but must be listed in the
 * `Tensors` template parameter. Any additional compute tags that hold a
 * `Tensor` can also be added to the `Tensors` template parameter. Finally,
 * `Variables` and other non-tensor compute tags can be listed in the
 * `NonTensorComputeTags` to facilitate observing. Note that the
 * `InertialCoordinates` are always observed.
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
template <size_t VolumeDim, typename... Tensors,
          typename... NonTensorComputeTags, typename ArraySectionIdTag>
class ObserveFields<VolumeDim, tmpl::list<Tensors...>,
                    tmpl::list<NonTensorComputeTags...>, ArraySectionIdTag>
    : public Event {
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
    static size_t upper_bound_on_size() { return sizeof...(Tensors); }
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

  /// \brief A list of block or group names on which to observe.
  ///
  /// Set to `All` to observe everywhere.
  struct BlocksToObserve {
    using type =
        Options::Auto<std::vector<std::string>, Options::AutoLabel::All>;
    static constexpr Options::String help = {
        "A list of block and group names on which to observe."};
  };

  using options =
      tmpl::list<SubfileName, CoordinatesFloatingPointType, FloatingPointTypes,
                 VariablesToObserve, BlocksToObserve, InterpolateToMesh>;

  static constexpr Options::String help =
      "Observe volume tensor fields.\n"
      "\n"
      "Writes volume quantities:\n"
      " * InertialCoordinates\n"
      " * Tensors listed in the 'VariablesToObserve' option\n";

  ObserveFields() = default;

  ObserveFields(
      const std::string& subfile_name,
      FloatingPointType coordinates_floating_point_type,
      const std::vector<FloatingPointType>& floating_point_types,
      const std::vector<std::string>& variables_to_observe,
      std::optional<std::vector<std::string>> active_block_or_block_groups = {},
      std::optional<Mesh<VolumeDim>> interpolation_mesh = {},
      const Options::Context& context = {});

  using compute_tags_for_observation_box =
      tmpl::list<Tensors..., NonTensorComputeTags...>;

  using return_tags = tmpl::list<>;
  using argument_tags = tmpl::list<::Tags::ObservationBox,
                                   ::Events::Tags::ObserverMesh<VolumeDim>>;

  template <typename DataBoxType, typename ComputeTagsList,
            typename Metavariables, typename ParallelComponent>
  void operator()(const ObservationBox<DataBoxType, ComputeTagsList>& box,
                  const Mesh<VolumeDim>& mesh,
                  Parallel::GlobalCache<Metavariables>& cache,
                  const ElementId<VolumeDim>& array_index,
                  const ParallelComponent* const component,
                  const ObservationValue& observation_value) const {
    if (not active_block(get<domain::Tags::Domain<VolumeDim>>(box),
                         array_index)) {
      return;
    }
    // Skip observation on elements that are not part of a section
    const std::optional<std::string> section_observation_key =
        observers::get_section_observation_key<ArraySectionIdTag>(box);
    if (not section_observation_key.has_value()) {
      return;
    }
    call_operator_impl(subfile_path_ + *section_observation_key,
                       variables_to_observe_, interpolation_mesh_, mesh, box,
                       cache, array_index, component, observation_value);
  }

  // We factor out the work into a static member function so it can  be shared
  // with other field observing events, like the one that deals with DG-subcell
  // where there are two grids. This is to avoid copy-pasting all of the code.
  template <typename DataBoxType, typename ComputeTagsList,
            typename Metavariables, typename ParallelComponent>
  static void call_operator_impl(
      const std::string& subfile_path,
      const std::unordered_map<std::string, FloatingPointType>&
          variables_to_observe,
      const std::optional<Mesh<VolumeDim>>& interpolation_mesh,
      const Mesh<VolumeDim>& mesh,
      const ObservationBox<DataBoxType, ComputeTagsList>& box,
      Parallel::GlobalCache<Metavariables>& cache,
      const ElementId<VolumeDim>& element_id,
      const ParallelComponent* const /*meta*/,
      const ObservationValue& observation_value) {
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
            std::decay_t<decltype(value(typename Tensors::type{}))>::size()...},
        0_st));

    const auto record_tensor_component_impl =
        [&components](DataVector&& tensor_component,
                      const FloatingPointType floating_point_type,
                      const std::string& component_name) {
          if (floating_point_type == FloatingPointType::Float) {
            components.emplace_back(component_name,
                                    std::vector<float>{tensor_component.begin(),
                                                       tensor_component.end()});
          } else {
            components.emplace_back(component_name,
                                    std::move(tensor_component));
          }
        };

    const auto record_tensor_components_impl =
        [&record_tensor_component_impl, &interpolant](
            const auto& tensor, const FloatingPointType floating_point_type,
            const std::string& tag_name) {
          using TensorType = std::decay_t<decltype(tensor)>;
          using VectorType = typename TensorType::type;
          for (size_t i = 0; i < tensor.size(); ++i) {
            auto tensor_component = interpolant.interpolate(tensor[i]);
            const std::string component_name =
                tag_name + tensor.component_suffix(i);
            if constexpr (std::is_same_v<VectorType, ComplexDataVector>) {
              record_tensor_component_impl(real(tensor_component),
                                           floating_point_type,
                                           "Re(" + component_name + ")");
              record_tensor_component_impl(imag(tensor_component),
                                           floating_point_type,
                                           "Im(" + component_name + ")");
            } else {
              record_tensor_component_impl(std::move(tensor_component),
                                           floating_point_type, component_name);
            }
          }
        };
    const auto record_tensor_components =
        [&box, &record_tensor_components_impl,
         &variables_to_observe](const auto tensor_tag_v) {
          using tensor_tag = tmpl::type_from<decltype(tensor_tag_v)>;
          const std::string tag_name = db::tag_name<tensor_tag>();
          if (const auto var_to_observe = variables_to_observe.find(tag_name);
              var_to_observe != variables_to_observe.end()) {
            const auto& tensor = get<tensor_tag>(box);
            if (not has_value(tensor)) {
              // This will only print a warning the first time it's called on a
              // node.
              [[maybe_unused]] static bool t =
                  ObserveFields::print_warning_about_optional<tensor_tag>();
              return;
            }
            const auto floating_point_type = var_to_observe->second;
            record_tensor_components_impl(value(tensor), floating_point_type,
                                          tag_name);
          }
        };
    EXPAND_PACK_LEFT_TO_RIGHT(record_tensor_components(tmpl::type_<Tensors>{}));

    const Parallel::ArrayComponentId array_component_id{
        std::add_pointer_t<ParallelComponent>{nullptr},
        Parallel::ArrayIndex<ElementId<VolumeDim>>{element_id}};
    ElementVolumeData element_volume_data{element_id, std::move(components),
                                          interpolation_mesh.value_or(mesh)};
    observers::ObservationId observation_id{observation_value.value,
                                            subfile_path + ".vol"};

    auto& local_observer = *Parallel::local_branch(
        Parallel::get_parallel_component<
            tmpl::conditional_t<Parallel::is_nodegroup_v<ParallelComponent>,
                                observers::ObserverWriter<Metavariables>,
                                observers::Observer<Metavariables>>>(cache));

    if constexpr (Parallel::is_nodegroup_v<ParallelComponent>) {
      // Send data to reduction observer writer (nodegroup)
      std::unordered_map<Parallel::ArrayComponentId,
                         std::vector<ElementVolumeData>>
          data_to_send{};
      data_to_send[array_component_id] =
          std::vector{std::move(element_volume_data)};
      Parallel::threaded_action<
          observers::ThreadedActions::ContributeVolumeDataToWriter>(
          local_observer, std::move(observation_id), array_component_id,
          subfile_path, std::move(data_to_send));
    } else {
      // Send data to volume observer
      Parallel::simple_action<observers::Actions::ContributeVolumeData>(
          local_observer, std::move(observation_id), subfile_path,
          array_component_id, std::move(element_volume_data));
    }
  }

  using observation_registration_tags = tmpl::list<::Tags::DataBox>;

  template <typename DbTagsList>
  std::optional<
      std::pair<observers::TypeOfObservation, observers::ObservationKey>>
  get_observation_type_and_key_for_registration(
      const db::DataBox<DbTagsList>& box) const {
    if (not active_block(db::get<domain::Tags::Domain<VolumeDim>>(box),
                         db::get<domain::Tags::Element<VolumeDim>>(box).id())) {
      return std::nullopt;
    }
    const std::optional<std::string> section_observation_key =
        observers::get_section_observation_key<ArraySectionIdTag>(box);
    if (not section_observation_key.has_value()) {
      return std::nullopt;
    }
    return {{observers::TypeOfObservation::Volume,
             observers::ObservationKey{
                 subfile_path_ + section_observation_key.value() + ".vol"}}};
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
    p | active_block_or_block_groups_;
    p | interpolation_mesh_;
  }

 private:
  template <typename Tag>
  static bool print_warning_about_optional() {
    Parallel::printf(
        "Warning: ObserveFields is trying to dump the tag %s "
        "but it is stored as a std::optional and has not been "
        "evaluated. This most commonly occurs when you are "
        "trying to either observe an analytic solution or errors when "
        "no analytic solution is available.\n",
        db::tag_name<Tag>());
    return false;
  }

  bool active_block(const Domain<VolumeDim>& domain,
                    const ElementId<VolumeDim>& element_id) const {
    if (not active_block_or_block_groups_.has_value()) {
      return true;
    }
    const std::unordered_set<std::string> block_names =
        domain::expand_block_groups_to_block_names(
            active_block_or_block_groups_.value(), domain.block_names(),
            domain.block_groups());
    return alg::found(block_names,
                      domain.blocks().at(element_id.block_id()).name());
  }

  std::string subfile_path_;
  std::unordered_map<std::string, FloatingPointType> variables_to_observe_{};
  std::optional<std::vector<std::string>> active_block_or_block_groups_{};
  std::optional<Mesh<VolumeDim>> interpolation_mesh_{};
};

template <size_t VolumeDim, typename... Tensors,
          typename... NonTensorComputeTags, typename ArraySectionIdTag>
ObserveFields<VolumeDim, tmpl::list<Tensors...>,
              tmpl::list<NonTensorComputeTags...>, ArraySectionIdTag>::
    ObserveFields(
        const std::string& subfile_name,
        const FloatingPointType coordinates_floating_point_type,
        const std::vector<FloatingPointType>& floating_point_types,
        const std::vector<std::string>& variables_to_observe,
        std::optional<std::vector<std::string>> active_block_or_block_groups,
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
      active_block_or_block_groups_(std::move(active_block_or_block_groups)),
      interpolation_mesh_(interpolation_mesh) {
  ASSERT(
      (... or (db::tag_name<Tensors>() == "InertialCoordinates")),
      "There is no tag with name 'InertialCoordinates' specified "
      "for the observer. Please make sure you specify a tag in the 'Tensors' "
      "list that has the 'db::tag_name()' 'InertialCoordinates'.");
  db::validate_selection<tmpl::list<Tensors...>>(variables_to_observe, context);
  variables_to_observe_["InertialCoordinates"] =
      coordinates_floating_point_type;
}

/// \cond
template <size_t VolumeDim, typename... Tensors,
          typename... NonTensorComputeTags, typename ArraySectionIdTag>
PUP::able::PUP_ID ObserveFields<VolumeDim, tmpl::list<Tensors...>,
                                tmpl::list<NonTensorComputeTags...>,
                                ArraySectionIdTag>::my_PUP_ID = 0;  // NOLINT
/// \endcond
}  // namespace Events
}  // namespace dg
