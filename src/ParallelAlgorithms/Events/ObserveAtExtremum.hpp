// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <pup.h>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/ObservationBox.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "IO/Observer/GetSectionObservationKey.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "IO/Observer/TypeOfObservation.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/String.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/Events/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/OptionalHelpers.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace Events {
/// @{
/*!
 * \brief Find the extremum of a Scalar<DataVector> over
 * all elements, as well as the value of other functions
 * at the location of that extremum.
 *
 *
 * Here is an example of an input file:
 *
 * \snippet Test_ObserveAtExtremum.cpp input_file_examples
 *
 * \par Array sections
 * This event supports sections (see `Parallel::Section`). Set the
 * `ArraySectionIdTag` template parameter to split up observations into subsets
 * of elements. The `observers::Tags::ObservationKey<ArraySectionIdTag>` must be
 * available in the DataBox. It identifies the section and is used as a suffix
 * for the path in the output file.
 */
template <typename ObservationValueTag, typename ObservableTensorTagsList,
          typename NonTensorComputeTagsList = tmpl::list<>,
          typename ArraySectionIdTag = void>
class ObserveAtExtremum;

template <typename ObservationValueTag, typename... ObservableTensorTags,
          typename... NonTensorComputeTags, typename ArraySectionIdTag>
class ObserveAtExtremum<ObservationValueTag,
                        tmpl::list<ObservableTensorTags...>,
                        tmpl::list<NonTensorComputeTags...>, ArraySectionIdTag>
    : public Event {
 private:
  /// Reduction data will contain the time, and either the maximum
  /// or the minimum of a function
  template <typename MinMaxFunctional>
  using ReductionData = Parallel::ReductionData<
      // Observation value
      Parallel::ReductionDatum<double, funcl::AssertEqual<>>,
      // Maximum of first component of a vector
      Parallel::ReductionDatum<std::vector<double>, MinMaxFunctional>>;

  /// Information about the scalar whose extremum we search for,
  /// the type of extremum, and the other tensors to observer at
  /// that extremum
  struct ObserveTensors {
    static constexpr Options::String help = {
        "The tensor to extremize, and other tensors to observe."};

    struct Name {
      using type = std::string;
      static constexpr Options::String help = {
          "The name of the scalar to extremize."};
    };

    struct ExtremumType {
      using type = std::string;
      static constexpr Options::String help = {
          "The type of extremum -- either Min or Max."};
    };

    struct AdditionalData {
      using type = std::vector<std::string>;
      static constexpr Options::String help = {
          "List of other tensors to observe at the extremum"};
    };

    using options = tmpl::list<Name, ExtremumType, AdditionalData>;

    ObserveTensors() = default;

    ObserveTensors(std::string in_scalar, std::string in_extremum_type,
                   std::vector<std::string> in_additional_data,
                   const Options::Context& context = {});

    std::string scalar_name;
    std::string extremum_type;
    std::vector<std::string> additional_data{};
  };

 public:
  /// The name of the subfile inside the HDF5 file
  struct SubfileName {
    using type = std::string;
    static constexpr Options::String help = {
        "The name of the subfile inside the HDF5 file without an extension and "
        "without a preceding '/'."};
  };
  /// The scalar to extremize, and other tensors to observe at extremum
  struct TensorsToObserve {
    using type = ObserveTensors;
    static constexpr Options::String help = {
        "Struct specifying the scalar to extremize, the type of extremum "
        "and other tensors to observe at that extremum."};
  };

  explicit ObserveAtExtremum(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ObserveAtExtremum);  // NOLINT

  using options = tmpl::list<SubfileName, TensorsToObserve>;

  static constexpr Options::String help =
      "Observe extremum of a scalar in the DataBox.\n"
      "\n"
      "Writes reduction quantities:\n"
      " * ObservationValueTag (e.g. Time or IterationId)\n"
      " * Extremum value of the desired scalar\n"
      " * Additional data at extremum\n";

  ObserveAtExtremum() = default;

  ObserveAtExtremum(std::string subfile_name,
                    ObserveTensors observe_tensors);

  using observed_reduction_data_tags = observers::make_reduction_data_tags<
      tmpl::list<ReductionData<funcl::Max<>>, ReductionData<funcl::Min<>>>>;

  using compute_tags_for_observation_box =
      tmpl::list<ObservableTensorTags..., NonTensorComputeTags...>;

  using argument_tags = tmpl::list<ObservationValueTag, ::Tags::ObservationBox>;

  template <typename ComputeTagsList, typename DataBoxType,
            typename Metavariables, size_t VolumeDim,
            typename ParallelComponent>
  void operator()(const typename ObservationValueTag::type& observation_value,
                  const ObservationBox<ComputeTagsList, DataBoxType>& box,
                  Parallel::GlobalCache<Metavariables>& cache,
                  const ElementId<VolumeDim>& array_index,
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
  std::string subfile_path_;
  std::string scalar_name_;
  std::string extremum_type_;
  std::vector<std::string> additional_tensor_names_{};
};
/// @}

/// \cond
template <typename ObservationValueTag, typename... ObservableTensorTags,
          typename... NonTensorComputeTags, typename ArraySectionIdTag>
ObserveAtExtremum<ObservationValueTag, tmpl::list<ObservableTensorTags...>,
                  tmpl::list<NonTensorComputeTags...>,
                  ArraySectionIdTag>::ObserveAtExtremum(CkMigrateMessage* msg)
    : Event(msg) {}

template <typename ObservationValueTag, typename... ObservableTensorTags,
          typename... NonTensorComputeTags, typename ArraySectionIdTag>
ObserveAtExtremum<ObservationValueTag, tmpl::list<ObservableTensorTags...>,
                  tmpl::list<NonTensorComputeTags...>, ArraySectionIdTag>::
    ObserveAtExtremum(std::string subfile_name,
                      ObserveTensors observe_tensors)
    : subfile_path_("/" + subfile_name),
      scalar_name_(std::move(observe_tensors.scalar_name)),
      extremum_type_(std::move(observe_tensors.extremum_type)),
      additional_tensor_names_(std::move(observe_tensors.additional_data)) {}

template <typename ObservationValueTag, typename... ObservableTensorTags,
          typename... NonTensorComputeTags, typename ArraySectionIdTag>
ObserveAtExtremum<ObservationValueTag, tmpl::list<ObservableTensorTags...>,
                  tmpl::list<NonTensorComputeTags...>, ArraySectionIdTag>::
    ObserveTensors::ObserveTensors(std::string in_scalar,
                                   std::string in_extremum_type,
                                   std::vector<std::string> in_additional_data,
                                   const Options::Context& context)
    : scalar_name(std::move(in_scalar)),
      extremum_type(std::move(in_extremum_type)),
      additional_data(std::move(in_additional_data)) {
  if (((scalar_name != db::tag_name<ObservableTensorTags>()) and ...)) {
    PARSE_ERROR(
        context, "Tensor '"
                     << scalar_name << "' is not known. Known tensors are: "
                     << ((db::tag_name<ObservableTensorTags>() + ",") + ...));
  }

  tmpl::for_each<tmpl::list<ObservableTensorTags...>>(
      [this, &context](auto tag_v) {
        using tag = tmpl::type_from<decltype(tag_v)>;
        const std::string tensor_name = db::tag_name<tag>();
        if (tensor_name == scalar_name) {
          if constexpr (tt::is_a_v<std::optional, typename tag::type>) {
            if (tag::type::value_type::rank() != 0) {
              PARSE_ERROR(context,
                          "ObserveAtExtremum can only observe scalars!");
            }
          } else if (tag::type::rank() != 0) {
            PARSE_ERROR(context, "ObserveAtExtremum can only observe scalars!");
          }
        }
      });

  if (extremum_type != "Max" and extremum_type != "Min") {
    PARSE_ERROR(context, "Extremum type " << extremum_type
                                          << " not recognized; use Max or Min");
  }
  for (const auto& tensor : additional_data) {
    if (((tensor != db::tag_name<ObservableTensorTags>()) and ...)) {
      PARSE_ERROR(context,
                  "Tensor '"
                      << tensor << "' is not known. Known tensors are: "
                      << ((db::tag_name<ObservableTensorTags>() + ",") + ...));
    }
  }
}

template <typename ObservationValueTag, typename... ObservableTensorTags,
          typename... NonTensorComputeTags, typename ArraySectionIdTag>
template <typename ComputeTagsList, typename DataBoxType,
          typename Metavariables, size_t VolumeDim, typename ParallelComponent>
void ObserveAtExtremum<ObservationValueTag, tmpl::list<ObservableTensorTags...>,
                       tmpl::list<NonTensorComputeTags...>, ArraySectionIdTag>::
operator()(const typename ObservationValueTag::type& observation_value,
           const ObservationBox<ComputeTagsList, DataBoxType>& box,
           Parallel::GlobalCache<Metavariables>& cache,
           const ElementId<VolumeDim>& array_index,
           const ParallelComponent* const /*meta*/) const {
  // Skip observation on elements that are not part of a section
  const std::optional<std::string> section_observation_key =
      observers::get_section_observation_key<ArraySectionIdTag>(box);
  if (not section_observation_key.has_value()) {
    return;
  }

  using tensor_tags = tmpl::list<ObservableTensorTags...>;

  // Vector that will contain the local extremum, and the value
  // of other tensors at that extremum
  std::vector<double> data_to_reduce{};
  // Vector containing a description of the data to be reduced.
  std::vector<std::string> legend{db::tag_name<ObservationValueTag>()};
  // Location of the local extremum
  size_t index_of_extremum = 0;
  // First, look for local extremum of desired scalar
  tmpl::for_each<tensor_tags>([this, &box, &data_to_reduce, &legend,
                               &index_of_extremum](auto tag_v) {
    using tag = tmpl::type_from<decltype(tag_v)>;
    const std::string tensor_name = db::tag_name<tag>();
    if (tensor_name == scalar_name_) {
      if (UNLIKELY(not has_value(get<tag>(box)))) {
        ERROR("Cannot observe a norm of '"
              << tensor_name
              << "' because it is a std::optional and wasn't able to be "
                 "computed. This can happen when you try to observe errors "
                 "without an analytic solution.");
      }
      const auto& scalar = value(get<tag>(box));
      const auto components = get<1>(scalar.get_vector_of_data());
      if (components.size() > 1) {
        ERROR("Extremum should be taken on a scalar, yet we have "
              << components.size() << " components in tensor " << tensor_name);
      }
      for (size_t i = 1; i < components[0].size(); i++) {
        if ((extremum_type_ == "Max" and
             (components[0][i] > components[0][index_of_extremum])) or
            (extremum_type_ == "Min" and
             (components[0][i] < components[0][index_of_extremum]))) {
          index_of_extremum = i;
        }
      }
      data_to_reduce.push_back(components[0][index_of_extremum]);
      if (extremum_type_ == "Max") {
        legend.push_back("Max(" + scalar_name_ + ")");
      } else {
        legend.push_back("Min(" + scalar_name_ + ")");
      }
    }
  });
  // Now get value of additional tensors at extremum
  tmpl::for_each<tensor_tags>([this, &box, &data_to_reduce, &legend,
                               &index_of_extremum](auto tag_v) {
    using tag = tmpl::type_from<decltype(tag_v)>;
    const std::string tensor_name = db::tag_name<tag>();
    for (size_t i = 0; i < additional_tensor_names_.size(); ++i)
      if (tensor_name == additional_tensor_names_[i]) {
        if (UNLIKELY(not has_value(get<tag>(box)))) {
          ERROR("Cannot observe a norm of '"
                << tensor_name
                << "' because it is a std::optional and wasn't able to be "
                   "computed. This can happen when you try to observe errors "
                   "without an analytic solution.");
        }
        const auto& tensor = value(get<tag>(box));
        const auto [component_names, components] = tensor.get_vector_of_data();
        for (size_t j = 0; j < components.size(); j++) {
          data_to_reduce.push_back(components[j][index_of_extremum]);
          if (components.size() > 1) {
            legend.push_back("At" + scalar_name_ + extremum_type_ + "(" +
                             tensor_name + "_" + component_names[j] + ")");
          } else {
            legend.push_back("At" + scalar_name_ + extremum_type_ + "(" +
                             tensor_name + ")");
          }
        }
      }
  });

  // Send data to reduction observer
  auto& local_observer = *Parallel::local_branch(
      Parallel::get_parallel_component<observers::Observer<Metavariables>>(
          cache));
  const std::string subfile_path_with_suffix =
      subfile_path_ + section_observation_key.value();

  if (extremum_type_ == "Max") {
    Parallel::simple_action<observers::Actions::ContributeReductionData>(
        local_observer,
        observers::ObservationId(observation_value,
                                 subfile_path_with_suffix + ".dat"),
        observers::ArrayComponentId{
            std::add_pointer_t<ParallelComponent>{nullptr},
            Parallel::ArrayIndex<ElementId<VolumeDim>>(array_index)},
        subfile_path_with_suffix, std::move(legend),
        ReductionData<funcl::Max<>>{static_cast<double>(observation_value),
                                    std::move(data_to_reduce)});
  } else {
    Parallel::simple_action<observers::Actions::ContributeReductionData>(
        local_observer,
        observers::ObservationId(observation_value,
                                 subfile_path_with_suffix + ".dat"),
        observers::ArrayComponentId{
            std::add_pointer_t<ParallelComponent>{nullptr},
            Parallel::ArrayIndex<ElementId<VolumeDim>>(array_index)},
        subfile_path_with_suffix, std::move(legend),
        ReductionData<funcl::Min<>>{static_cast<double>(observation_value),
                                    std::move(data_to_reduce)});
  }
}

template <typename ObservationValueTag, typename... ObservableTensorTags,
          typename... NonTensorComputeTags, typename ArraySectionIdTag>
void ObserveAtExtremum<ObservationValueTag, tmpl::list<ObservableTensorTags...>,
                       tmpl::list<NonTensorComputeTags...>,
                       ArraySectionIdTag>::pup(PUP::er& p) {
  Event::pup(p);
  p | subfile_path_;
  p | scalar_name_;
  p | extremum_type_;
  p | additional_tensor_names_;
}

template <typename ObservationValueTag, typename... ObservableTensorTags,
          typename... NonTensorComputeTags, typename ArraySectionIdTag>
PUP::able::PUP_ID
    ObserveAtExtremum<ObservationValueTag, tmpl::list<ObservableTensorTags...>,
                      tmpl::list<NonTensorComputeTags...>,
                      ArraySectionIdTag>::my_PUP_ID = 0;  // NOLINT
/// \endcond
}  // namespace Events
