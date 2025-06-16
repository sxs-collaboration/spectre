// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <pup.h>
#include <string>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/ObservationBox.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Particles/MonteCarlo/ManagerComponent.hpp"
#include "Evolution/Particles/MonteCarlo/Packet.hpp"
#include "Evolution/Particles/MonteCarlo/Tags.hpp"
#include "IO/Observer/GetSectionObservationKey.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "IO/Observer/TypeOfObservation.hpp"
#include "Options/String.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/Reduction.hpp"
#include "Parallel/TypeTraits.hpp"
#include "ParallelAlgorithms/Events/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/OptionalHelpers.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace Events::MonteCarlo {

template <size_t NeutrinoSpecies, typename ObservableTensorTagsList,
          typename NonTensorComputeTagsList = tmpl::list<>,
          typename ArraySectionIdTag = void, typename OptionName = void>
class GlobalReductions;

namespace Actions {

/// Action taken by each element after reduction. It is called from
/// PostReductionManagerAction, which sends global information back
/// to individual elements.
///
/// DataBox changes:
/// - Modifies:
///   * Particles::MonteCarlo::Tags::
///       MinimumPacketEnergyAtEmission<NeutrinoSpecies>
///   * Particles::MonteCarlo::Tags::PacketsOnElement
///   * Particles::MonteCarlo::Tags::RandomNumberGenerator
template <size_t NeutrinoSpecies>
struct PostReductionComponentAction {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(db::DataBox<DbTags>& box,
                    Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const std::array<double, NeutrinoSpecies>&
                        updated_minimum_packet_energy) {
    // Update desired packet energy in low-density regions if we have too many
    // packets, or too few.
    db::mutate<Particles::MonteCarlo::Tags::MinimumPacketEnergyAtEmission<
                   NeutrinoSpecies>,
               Particles::MonteCarlo::Tags::PacketsOnElement,
               Particles::MonteCarlo::Tags::RandomNumberGenerator>(
        [&updated_minimum_packet_energy](
            const gsl::not_null<std::array<double, NeutrinoSpecies>*>
                minimum_packet_energy,
            const gsl::not_null<std::vector<Particles::MonteCarlo::Packet>*>
                packets,
            const gsl::not_null<std::mt19937*> random_number_generator) {
          std::array<bool, NeutrinoSpecies> packet_energy_is_increased;
          std::array<double, NeutrinoSpecies> survival_probability;
          bool need_packet_update = false;
          for (size_t d = 0; d < NeutrinoSpecies; d++) {
            gsl::at(packet_energy_is_increased, d) = false;
            gsl::at(survival_probability, d) = 1.0;
            if (gsl::at(updated_minimum_packet_energy, d) >
                gsl::at((*minimum_packet_energy), d)) {
              gsl::at(packet_energy_is_increased, d) = true;
              need_packet_update = true;
              gsl::at(survival_probability, d) =
                  gsl::at((*minimum_packet_energy), d) /
                  gsl::at(updated_minimum_packet_energy, d);
            }
            gsl::at((*minimum_packet_energy), d) =
                gsl::at(updated_minimum_packet_energy, d);
          }
          if (need_packet_update) {
            // If we increase the minimum energy of packets, we also resample
            // existing packets
            std::uniform_real_distribution<double> rng_uniform_zero_to_one(0.0,
                                                                           1.0);
            size_t n_packets = packets->size();
            for (size_t p = 0; p < n_packets; p++) {
              const size_t& species = (*packets)[p].species;
              if (gsl::at(packet_energy_is_increased, species)) {
                // Two possibilities here: either we delete the packet, or we
                // reweight it to compensate for the removed packets
                if (rng_uniform_zero_to_one(*random_number_generator) <
                    gsl::at(survival_probability, species)) {
                  (*packets)[p].number_of_neutrinos *=
                      1.0 / gsl::at(survival_probability, species);
                } else {
                  std::swap((*packets)[p], (*packets)[n_packets - 1]);
                  packets->pop_back();
                  p--;
                  n_packets--;
                }
              }
            }
          }
        },
        make_not_null(&box));
  }
};

/// Action that gather MC information from global reduction and modify
/// global simulation settings as needed.
/// So far, this monitors the number of packets and adapt the desired
/// energy of emitted packets.
///
/// DataBox changes:
/// - Modifies:
///   * Particles::MonteCarlo::Tags::
///       MinimumPacketEnergyAtEmission<NeutrinoSpecies>
template <size_t NeutrinoSpecies>
struct PostReductionManagerAction {
  using const_global_cache_tags = tmpl::list<
      Particles::MonteCarlo::Tags::MonteCarloOptions<NeutrinoSpecies>>;

  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(db::DataBox<DbTags>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const double& /*observation_id*/,
                    const std::array<size_t, NeutrinoSpecies>& n_packets) {
    const auto mc_options = db::get<
        Particles::MonteCarlo::Tags::MonteCarloOptions<NeutrinoSpecies>>(box);
    const auto& initial_packet_energy = mc_options.get_initial_packet_energy();
    const size_t& desired_packets_per_species =
        mc_options.get_desired_packets_per_species();
    std::array<double, NeutrinoSpecies> updated_minimum_packet_energy;
    for (size_t d = 0; d < NeutrinoSpecies; d++) {
      gsl::at(updated_minimum_packet_energy, d) = 0.0;
    }

    // Update desired packet energy in low-density regions if we have too many
    // packets, or too few.
    db::mutate<Particles::MonteCarlo::Tags::MinimumPacketEnergyAtEmission<
        NeutrinoSpecies>>(
        [&desired_packets_per_species, &n_packets, &initial_packet_energy,
         &updated_minimum_packet_energy](
            const gsl::not_null<std::array<double, NeutrinoSpecies>*>
                minimum_packet_energy) {
          for (size_t d = 0; d < NeutrinoSpecies; d++) {
            if (desired_packets_per_species < gsl::at(n_packets, d)) {
              gsl::at((*minimum_packet_energy), d) *= 1.0 / 0.9;
            } else if (0.81 * desired_packets_per_species >
                       gsl::at(n_packets, d)) {
              gsl::at((*minimum_packet_energy), d) =
                  std::max(gsl::at((*minimum_packet_energy), d) * 0.9,
                           gsl::at(initial_packet_energy, d));
            }
            gsl::at(updated_minimum_packet_energy, d) =
                gsl::at((*minimum_packet_energy), d);
          }
        },
        make_not_null(&box));
    Parallel::printf(
        "Using packet energies %f %f %f\n", updated_minimum_packet_energy[0],
        updated_minimum_packet_energy[1], updated_minimum_packet_energy[2]);
    auto& reduction_target_proxy = Parallel::get_parallel_component<
        typename Metavariables::dg_element_array>(cache);
    Parallel::simple_action<PostReductionComponentAction<NeutrinoSpecies>>(
        reduction_target_proxy, updated_minimum_packet_energy);
  }
};

}  // namespace Actions

template <size_t NeutrinoSpecies, typename... ObservableTensorTags,
          typename... NonTensorComputeTags, typename ArraySectionIdTag,
          typename OptionName>
class GlobalReductions<NeutrinoSpecies, tmpl::list<ObservableTensorTags...>,
                       tmpl::list<NonTensorComputeTags...>, ArraySectionIdTag,
                       OptionName> : public Event {
 private:
  using ReductionData = Parallel::ReductionData<
      // Observation value
      Parallel::ReductionDatum<double, funcl::AssertEqual<>>,
      Parallel::ReductionDatum<std::array<size_t, NeutrinoSpecies>,
                               funcl::Plus<>>>;

 public:
  static std::string name() { return "GlobalReductionsMonteCarlo"; }

  /// The name of the subfile inside the HDF5 file
  struct SubfileName {
    using type = std::string;
    static constexpr Options::String help = {
        "The name of the subfile inside the HDF5 file without an extension and "
        "without a preceding '/'."};
  };

  explicit GlobalReductions(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(GlobalReductions);  // NOLINT

  using options = tmpl::list<SubfileName>;

  static constexpr Options::String help = "Global reductions for MonteCarlo.\n";

  GlobalReductions() = default;
  explicit GlobalReductions(const std::string& subfile_name);

  using observed_reduction_data_tags =
      observers::make_reduction_data_tags<tmpl::list<ReductionData>>;

  using compute_tags_for_observation_box =
      tmpl::list<ObservableTensorTags..., NonTensorComputeTags...>;

  using return_tags = tmpl::list<>;
  using argument_tags = tmpl::list<::Tags::ObservationBox>;

  template <typename ComputeTagsList, typename DataBoxType,
            typename Metavariables, size_t VolumeDim,
            typename ParallelComponent>
  void operator()(const ObservationBox<ComputeTagsList, DataBoxType>& box,
                  Parallel::GlobalCache<Metavariables>& cache,
                  const ElementId<VolumeDim>& array_index,
                  const ParallelComponent* /*meta*/,
                  const ObservationValue& observation_value) const;

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

  bool needs_evolved_variables() const override { return false; }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

 private:
  std::string subfile_path_;
};
/// @}

/// \cond
template <size_t NeutrinoSpecies, typename... ObservableTensorTags,
          typename... NonTensorComputeTags, typename ArraySectionIdTag,
          typename OptionName>
GlobalReductions<NeutrinoSpecies, tmpl::list<ObservableTensorTags...>,
                 tmpl::list<NonTensorComputeTags...>, ArraySectionIdTag,
                 OptionName>::GlobalReductions(CkMigrateMessage* msg)
    : Event(msg) {}

template <size_t NeutrinoSpecies, typename... ObservableTensorTags,
          typename... NonTensorComputeTags, typename ArraySectionIdTag,
          typename OptionName>
GlobalReductions<NeutrinoSpecies, tmpl::list<ObservableTensorTags...>,
                 tmpl::list<NonTensorComputeTags...>, ArraySectionIdTag,
                 OptionName>::GlobalReductions(const std::string& subfile_name)
    : subfile_path_("/" + subfile_name) {}

template <size_t NeutrinoSpecies, typename... ObservableTensorTags,
          typename... NonTensorComputeTags, typename ArraySectionIdTag,
          typename OptionName>
template <typename ComputeTagsList, typename DataBoxType,
          typename Metavariables, size_t VolumeDim, typename ParallelComponent>
void GlobalReductions<NeutrinoSpecies, tmpl::list<ObservableTensorTags...>,
                      tmpl::list<NonTensorComputeTags...>, ArraySectionIdTag,
                      OptionName>::
operator()(const ObservationBox<ComputeTagsList, DataBoxType>& box,
           Parallel::GlobalCache<Metavariables>& cache,
           const ElementId<VolumeDim>& array_index,
           const ParallelComponent* const /*meta*/,
           const ObservationValue& observation_value) const {
  // Skip observation on elements that are not part of a section
  const std::optional<std::string> section_observation_key =
      observers::get_section_observation_key<ArraySectionIdTag>(box);
  if (not section_observation_key.has_value()) {
    return;
  }

  const auto& packet_list =
      (get<Particles::MonteCarlo::Tags::PacketsOnElement>(box));
  std::array<size_t, NeutrinoSpecies> packets_per_species;
  for (size_t s = 0; s < NeutrinoSpecies; s++) {
    packets_per_species[s] = 0;
  }
  for (auto& packet : packet_list) {
    packets_per_species[packet.species]++;
  }

  // Concatenate the legend info together.
  const std::vector<std::string> legend{
    observation_value.name, "PacketsPerSpecies_0",
    "PacketsPerSpecies_1", "PacketsPerSpecies_2"};

  const std::string subfile_path_with_suffix =
      subfile_path_ + section_observation_key.value();
  ReductionData reduction_data{observation_value.value, packets_per_species};
  auto my_proxy =
      Parallel::get_parallel_component<ParallelComponent>(cache)[array_index];
  auto& reduction_target_proxy = Parallel::get_parallel_component<
      Particles::MonteCarlo::ManagerComponent<Metavariables>>(cache);
  Parallel::contribute_to_reduction<
      Actions::PostReductionManagerAction<NeutrinoSpecies>>(
      std::move(reduction_data), my_proxy, reduction_target_proxy);
}

template <size_t NeutrinoSpecies, typename... ObservableTensorTags,
          typename... NonTensorComputeTags, typename ArraySectionIdTag,
          typename OptionName>
void GlobalReductions<NeutrinoSpecies, tmpl::list<ObservableTensorTags...>,
                      tmpl::list<NonTensorComputeTags...>, ArraySectionIdTag,
                      OptionName>::pup(PUP::er& p) {
  Event::pup(p);
  p | subfile_path_;
}

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
template <size_t NeutrinoSpecies, typename... ObservableTensorTags,
          typename... NonTensorComputeTags, typename ArraySectionIdTag,
          typename OptionName>
PUP::able::PUP_ID
    GlobalReductions<NeutrinoSpecies, tmpl::list<ObservableTensorTags...>,
                     tmpl::list<NonTensorComputeTags...>, ArraySectionIdTag,
                     OptionName>::my_PUP_ID = 0;
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)
/// \endcond

}  // namespace Events::MonteCarlo
