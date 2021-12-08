// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <optional>
#include <pup.h>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/ObservationBox.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/IdPair.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/ElementLogicalCoordinates.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Structure/BlockId.hpp"
#include "IO/Observer/GetSectionObservationKey.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "IO/Observer/TypeOfObservation.hpp"
#include "NumericalAlgorithms/Interpolation/IrregularInterpolant.hpp"
#include "Options/Options.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Time/Tags.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/OptionalHelpers.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace dg::Events {
/// \cond
template <size_t Dim, typename ObservationValueTag,
          typename ObservableTensorTagsList,
          typename NonTensorComputeTagsList = tmpl::list<>,
          typename ArraySectionIdTag = void>
class ObserveAtPoint;
/// \endcond

/*!
 * \brief Interpolate tensors to a user-specified point and write the values to
 * disk
 *
 * The interpolated values are an easy way to compare simulation data on
 * different grids, or from different codes. The `ObservableTensorTags` are the
 * set of tensors in the DataBox that are available for observation. The user
 * can select a subset of these tensors, and the coordinates of the point to
 * interpolate to.
 *
 * To observe data at multiple points the user can specify multiple instances of
 * this event in the input file. This way, each instance of this event only
 * depends on data from the single element that the point is on, or the few
 * elements that share the point if it is on the interface of elements with
 * Gauss-Lobatto grids.
 *
 * When the domain is time dependent, the `domain::Tags::FunctionsOfTime` must
 * be up to date when this event is invoked.
 *
 * \par Array sections
 * This event supports sections (see `Parallel::Section`). Set the
 * `ArraySectionIdTag` template parameter to split up observations into subsets
 * of elements. The `observers::Tags::ObservationKey<ArraySectionIdTag>` must be
 * available in the DataBox. It identifies the section and is used as a suffix
 * for the path in the output file.
 */
template <size_t Dim, typename ObservationValueTag,
          typename... ObservableTensorTags, typename NonTensorComputeTagsList,
          typename ArraySectionIdTag>
class ObserveAtPoint<Dim, ObservationValueTag,
                     tmpl::list<ObservableTensorTags...>,
                     NonTensorComputeTagsList, ArraySectionIdTag>
    : public Event {
 private:
  using ReductionData = Parallel::ReductionData<
      // Observation value (e.g. time or iteration ID)
      Parallel::ReductionDatum<double, funcl::AssertEqual<>>,
      // Number of contributing elements. This is usually one, but can be larger
      // than one for Gauss-Lobatto grids when the user-specified point is on
      // an element boundary.
      Parallel::ReductionDatum<size_t, funcl::Plus<>>,
      // Interpolated value at the user-specified point for all selected tensor
      // components. We take the mean over all contributing elements.
      Parallel::ReductionDatum<
          std::vector<double>, funcl::ElementWise<funcl::Plus<>>,
          funcl::ElementWise<funcl::Divides<>>, std::index_sequence<1>>>;

 public:
  struct SubfileName {
    using type = std::string;
    static constexpr Options::String help = {
        "The name of the subfile inside the HDF5 file without an extension and "
        "without a preceding '/'."};
  };
  struct TensorsToObserve {
    using type = std::vector<std::string>;
    static constexpr Options::String help = {
        "List the names of tensors to interpolate to the specified point."};
  };
  struct Coordinates {
    using type = std::array<double, Dim>;
    static constexpr Options::String help = {
        "The coordinates of the point to interpolate to."};
  };

  /// \cond
  explicit ObserveAtPoint(CkMigrateMessage* msg) : Event(msg) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ObserveAtPoint);  // NOLINT
  /// \endcond

  using options = tmpl::list<SubfileName, Coordinates, TensorsToObserve>;
  static constexpr Options::String help =
      "Observe tensors interpolated to the specified point.\n";

  ObserveAtPoint() = default;
  ObserveAtPoint(const std::string& subfile_name,
                 std::array<double, Dim> coords,
                 std::vector<std::string> tensor_names,
                 const Options::Context& context = {})
      : subfile_path_("/" + subfile_name),
        coords_(std::move(coords)),
        tensor_names_(std::move(tensor_names)) {
    for (const auto& name : tensor_names_) {
      if (((name != db::tag_name<ObservableTensorTags>()) and ...)) {
        PARSE_ERROR(context, "No tensor with name '"
                                 << name << "' available to observe.");
      }
    }
  }

  using observed_reduction_data_tags =
      observers::make_reduction_data_tags<tmpl::list<ReductionData>>;

  using compute_tags_for_observation_box =
      tmpl::push_back<NonTensorComputeTagsList, ObservableTensorTags...>;

  using argument_tags = tmpl::list<ObservationValueTag, ::Tags::ObservationBox>;

  template <typename DataBoxType, typename ComputeTagsList,
            typename Metavariables, typename ParallelComponent>
  void operator()(const typename ObservationValueTag::type& observation_value,
                  const ObservationBox<DataBoxType, ComputeTagsList>& box,
                  Parallel::GlobalCache<Metavariables>& cache,
                  const ElementId<Dim>& element_id,
                  const ParallelComponent* const /*meta*/) const;

  using observation_registration_tags = tmpl::list<::Tags::DataBox>;

  template <typename DbTagsList>
  std::optional<
      std::pair<observers::TypeOfObservation, observers::ObservationKey>>
  get_observation_type_and_key_for_registration(
      const db::DataBox<DbTagsList>& box) const {
    const std::optional<std::string> section_observation_key =
        observers::get_section_observation_key<ArraySectionIdTag>(box);
    if (not section_observation_key.has_value()) {
      return std::nullopt;
    }
    // If the domain is static we can get away with registering only the
    // element(s) that contain the specified point at the time of registration
    if (not domain_is_time_dependent(db::get<domain::Tags::Domain<Dim>>(box))) {
      // This registration function is sometimes instantiated with an empty
      // DataBox
      if constexpr (db::tag_is_retrievable_v<domain::Tags::Element<Dim>,
                                             db::DataBox<DbTagsList>>) {
        const auto element_logical_coords = get_element_logical_coords(box);
        if (not element_logical_coords.has_value()) {
          return std::nullopt;
        }
      } else {
        ERROR(
            "Invoking registration with a DataBox that is missing "
            "'domain::Tags::Element'. Make sure the registration is invoked "
            "after initialization.");
      }
    }
    return {{observers::TypeOfObservation::Reduction,
             observers::ObservationKey(
                 subfile_path_ + section_observation_key.value() + ".dat")}};
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
  void pup(PUP::er& p) override;

 private:
  static bool domain_is_time_dependent(const Domain<Dim>& domain) {
    return alg::any_of(domain.blocks(), [](const auto& block) {
      return block.is_time_dependent();
    });
  }

  template <typename DataBoxType>
  std::optional<tnsr::I<DataVector, Dim, Frame::ElementLogical>>
  get_element_logical_coords(const DataBoxType& box) const {
    const auto& element_id = get<domain::Tags::Element<Dim>>(box).id();
    // The coordinates of the single point, as a tensor and stored in a
    // DataVector to comply with the interface used below
    tnsr::I<DataVector, Dim> coords_as_tensor{size_t{1}};
    for (size_t i = 0; i < Dim; ++i) {
      coords_as_tensor.get(i) = gsl::at(coords_, i);
    }
    // Note: If the point is on a block boundary, only the block with the lowest
    // block ID is returned here currently.
    const auto block_logical_coords = [*this, &coords_as_tensor, &box]() {
      const auto& domain = get<domain::Tags::Domain<Dim>>(box);
      if (domain_is_time_dependent(domain)) {
        if constexpr (db::tag_is_retrievable_v<domain::Tags::FunctionsOfTime,
                                               DataBoxType>) {
          return ::block_logical_coordinates(
              domain, coords_as_tensor, get<::Tags::Time>(box),
              get<domain::Tags::FunctionsOfTime>(box));
        } else {
          ERROR(
              "The domain is time-dependent, but no functions of time are "
              "available. If you intend to use a time-dependent domain, please "
              "add 'domain::Tags::FunctionsOfTime' to the GlobalCache.");
        }
      } else {
        return ::block_logical_coordinates(domain, coords_as_tensor);
      }
    }();
    // The subscript [0] refers to the single point we are looking at here
    if (not block_logical_coords[0].has_value()) {
      throw std::runtime_error{"The point " + get_output(coords_) +
                               " in an 'ObserveAtPoint' event is outside the "
                               "computational domain."};
    }
    const auto element_logical_coords =
        ::element_logical_coordinates({element_id}, block_logical_coords);
    const auto found_element_id = element_logical_coords.find(element_id);
    if (found_element_id == element_logical_coords.end()) {
      return std::nullopt;
    } else {
      return found_element_id->second.element_logical_coords;
    }
  }

  std::string subfile_path_;
  std::array<double, Dim> coords_{};
  std::vector<std::string> tensor_names_{};
};

/// \cond
template <size_t Dim, typename ObservationValueTag,
          typename... ObservableTensorTags, typename NonTensorComputeTagsList,
          typename ArraySectionIdTag>
template <typename DataBoxType, typename ComputeTagsList,
          typename Metavariables, typename ParallelComponent>
void ObserveAtPoint<Dim, ObservationValueTag,
                    tmpl::list<ObservableTensorTags...>,
                    NonTensorComputeTagsList, ArraySectionIdTag>::
operator()(const typename ObservationValueTag::type& observation_value,
           const ObservationBox<DataBoxType, ComputeTagsList>& box,
           Parallel::GlobalCache<Metavariables>& cache,
           const ElementId<Dim>& element_id,
           const ParallelComponent* const /*meta*/) const {
  // Skip observation on elements that are not part of a section
  const std::optional<std::string> section_observation_key =
      observers::get_section_observation_key<ArraySectionIdTag>(box);
  if (not section_observation_key.has_value()) {
    return;
  }
  using tensor_tags = tmpl::list<ObservableTensorTags...>;
  auto& local_observer =
      *Parallel::get_parallel_component<observers::Observer<Metavariables>>(
           cache)
           .ckLocalBranch();
  const std::string subfile_path_with_suffix =
      subfile_path_ + section_observation_key.value();

  // Construct the legend
  std::vector<std::string> legend{db::tag_name<ObservationValueTag>(),
                                  "NumContributingElements"};
  tmpl::for_each<tensor_tags>([this, &legend](auto tag_v) {
    using tag = tmpl::type_from<decltype(tag_v)>;
    const std::string tensor_name = db::tag_name<tag>();
    for (size_t tensor_index = 0; tensor_index < tensor_names_.size();
         ++tensor_index) {
      if (tensor_name != tensor_names_[tensor_index]) {
        continue;
      }
      using TensorType = std::decay_t<decltype(value(typename tag::type{}))>;
      for (size_t component = 0; component < TensorType::size(); ++component) {
        legend.push_back(tensor_name + TensorType::component_suffix(component));
      }
    }
  });

  // Check if the point is in the domain
  const bool is_time_dep =
      domain_is_time_dependent(get<domain::Tags::Domain<Dim>>(box));
  std::optional<tnsr::I<DataVector, Dim, Frame::ElementLogical>>
      element_logical_coords{};
  try {
    element_logical_coords = get_element_logical_coords(box);
  } catch (std::runtime_error& /*err*/) {
    if (is_time_dep) {
      // When the domain is time-dependent it's ok if the point is currently
      // outside the domain. We contribute NaN to the reduction so it's clear
      // that we didn't observe anything.
      Parallel::simple_action<observers::Actions::ContributeReductionData>(
          local_observer,
          observers::ObservationId(observation_value,
                                   subfile_path_with_suffix + ".dat"),
          observers::ArrayComponentId{
              std::add_pointer_t<ParallelComponent>{nullptr},
              Parallel::ArrayIndex<ElementId<Dim>>(element_id)},
          subfile_path_with_suffix, std::move(legend),
          ReductionData{
              static_cast<double>(observation_value), size_t{1},
              std::vector<double>(legend.size() - 2,
                                  std::numeric_limits<double>::quiet_NaN())});
      return;
    } else {
      // When the domain is static it's an error if the point is outside
      throw;
    }
  }

  // Check if the point is in this element
  if (not element_logical_coords.has_value()) {
    // The point is in the domain, but outside this element
    if (is_time_dep) {
      // Contribute nothing to the reduction
      Parallel::simple_action<observers::Actions::ContributeReductionData>(
          local_observer,
          observers::ObservationId(observation_value,
                                   subfile_path_with_suffix + ".dat"),
          observers::ArrayComponentId{
              std::add_pointer_t<ParallelComponent>{nullptr},
              Parallel::ArrayIndex<ElementId<Dim>>(element_id)},
          subfile_path_with_suffix, std::move(legend),
          ReductionData{static_cast<double>(observation_value), size_t{0},
                        std::vector<double>(legend.size() - 2, 0.)});
      return;
    } else {
      // We haven't even registered the element, so we can return right here
      // without contributing to the reduction.
      return;
    }
  }

  // Construct the interpolant to the user-specified point
  const intrp::Irregular<Dim> interpolant{get<domain::Tags::Mesh<Dim>>(box),
                                          *element_logical_coords};

  // Each component of the user-specified tensors, interpolated to the point
  std::vector<double> interpolated_data{};

  // Loop over ObservableTensorTags and see if it was requested to be observed
  // before retrieving it from the box. This approach allows us to delay
  // evaluating any compute tags until they're actually needed for observing.
  tmpl::for_each<tensor_tags>(
      [this, &box, &interpolant, &interpolated_data](auto tag_v) {
        using tag = tmpl::type_from<decltype(tag_v)>;
        const std::string tensor_name = db::tag_name<tag>();
        for (size_t tensor_index = 0; tensor_index < tensor_names_.size();
             ++tensor_index) {
          if (tensor_name != tensor_names_[tensor_index]) {
            continue;
          }
          if (UNLIKELY(not has_value(get<tag>(box)))) {
            ERROR("Cannot observe '"
                  << tensor_name
                  << "' because it is a std::optional and wasn't able to be "
                     "computed. This can happen when you try to observe errors "
                     "without an analytic solution.");
          }
          const auto& tensor = value(get<tag>(box));
          for (size_t component = 0; component < tensor.size(); ++component) {
            // The subscript [0] refers to the single point in the DataVector
            interpolated_data.push_back(
                interpolant.interpolate(tensor[component])[0]);
          }
        }
      });

  // Send data to reduction observer
  Parallel::simple_action<observers::Actions::ContributeReductionData>(
      local_observer,
      observers::ObservationId(observation_value,
                               subfile_path_with_suffix + ".dat"),
      observers::ArrayComponentId{
          std::add_pointer_t<ParallelComponent>{nullptr},
          Parallel::ArrayIndex<ElementId<Dim>>(element_id)},
      subfile_path_with_suffix, std::move(legend),
      ReductionData{static_cast<double>(observation_value), size_t{1},
                    std::move(interpolated_data)});
}

template <size_t Dim, typename ObservationValueTag,
          typename... ObservableTensorTags, typename NonTensorComputeTagsList,
          typename ArraySectionIdTag>
void ObserveAtPoint<
    Dim, ObservationValueTag, tmpl::list<ObservableTensorTags...>,
    NonTensorComputeTagsList, ArraySectionIdTag>::pup(PUP::er& p) {
  Event::pup(p);
  p | subfile_path_;
  p | coords_;
  p | tensor_names_;
}

template <size_t Dim, typename ObservationValueTag,
          typename... ObservableTensorTags, typename NonTensorComputeTagsList,
          typename ArraySectionIdTag>
PUP::able::PUP_ID ObserveAtPoint<
    Dim, ObservationValueTag, tmpl::list<ObservableTensorTags...>,
    NonTensorComputeTagsList, ArraySectionIdTag>::my_PUP_ID = 0;  // NOLINT
/// \endcond
}  // namespace dg::Events
