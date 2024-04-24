// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCoefficients.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCollocation.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshTransform.hpp"
#include "Options/Context.hpp"
#include "Options/ParseError.hpp"
#include "Options/String.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

/// \cond
template <size_t Dim>
class Mesh;
namespace Frame {
struct Inertial;
}  // namespace Frame
/// \endcond

namespace Cce {
namespace Events {
namespace detail {
template <typename Tag>
std::string name() {
  if constexpr (std::is_same_v<Tag, Tags::ComplexInertialRetardedTime>) {
    return db::tag_name<Tags::InertialRetardedTime>();
  } else {
    return db::tag_name<Tag>();
  }
}
}  // namespace detail

/*!
 * \brief Event to observe fields/variables in a characteristic evolution.
 *
 * \details Similar to `dg::Events::ObserveFields`, this event will write volume
 * data from the characteristic domain to disk when triggered. However, there
 * are several differences which are important to highlight.
 *
 * First is the fields themselves. The DG event takes the fields to observe as
 * template parameters because the event must work with many evolution systems.
 * However, since this event is specific to the characteristic evolution system,
 * we can hardcode the list of fields that are available to observe. The fields
 * available to observe are the following tags along with their first and second
 * `Cce::Tags::Dy` derivatives (see `Cce::Tags::Dy` for a definition of `y`):
 *
 * - `Cce::Tags::BondiBeta`
 * - `Cce::Tags::BondiU`
 * - `Cce::Tags::BondiQ`
 * - `Cce::Tags::BondiW`
 * - `Cce::Tags::BondiH` (no second derivative)
 * - `Cce::Tags::BondiJ`
 * - `Cce::Tags::Du<Cce::Tags::BondiJ>`
 *
 * Some more fields to observe are:
 *
 * - `Cce::Tags::ComplexInertialRetardedTime`
 * - `Cce::Tags::OneMinusY`
 * - `Cce::Tags::BondiR`
 * - `Cce::Tags::EthRDividedByR`
 * - `Cce::Tags::DuRDividedByR`
 *
 * The main reason that this event is separate from the DG one is because this
 * event writes modal data over the sphere for every radial grid point, while
 * the DG event writes nodal data. Every tag above is a
 * `Scalar<SpinWeighted<ComplexDataVector, Spin>>` for some `Spin`. While this
 * data itself is in nodal form, it is more convenient to transform to modal
 * data and decompose in spherical harmonics before writing. This means
 * our typical way of writing/storing volume data won't work.
 *
 * This event writes its data in the following structure in the H5 file:
 * `/Cce/VolumeData/TagName/CompactifiedRadius_X.dat`. Every field that is
 * observed will get its own subgroup called `TagName`. In this subgroup, there
 * will be N files corresponding to N radial grid points named
 * `CompactifiedRadius_X.dat` where `X` here will range from 0 to `N-1`. We call
 * these compactified radii because for a more "physical" radius, it goes to
 * infinity at future-null infinity and we can't write that in a file. Instead,
 * these N files will correspond to the compactified coordinate $y = 1 - 2R/r$
 * where $r$ is your coordinate radius and $R$ is the coordinate radius of your
 * worldtube. Each file will hold the modal data for that radial grid point. It
 * is recommended to always dump the quantity `Cce::Tags::OneMinusY` so the
 * values of the compactified coordinates are available as well.
 *
 * There are two notable exceptions to this format. One is
 * `Cce::Tags::ComplexInertialRetardedTime`. The quantity we are actually
 * interested in is `Cce::Tags::InertialRetardedTime` which is real and only
 * defined once for every direction $\theta,\phi$ (meaning it does not have
 * different values at the different radial grid points). However, we use
 * `Cce::Tags::ComplexInertialRetardedTime` because it has the same data type as
 * the other tags which makes the internals of the class simpler. The imaginary
 * part of this `ComplexDataVector` is set to zero. This quantity will be stored
 * in a subfile named `/Cce/VolumeData/InertialRetardedTime.dat` as a single
 * modal set of data so we don't repeat it N times.
 *
 * The second is `Cce::Tags::OneMinusY`. Even though this quantity is stored
 * as a `Scalar<SpinWeighted<ComplexDataVector, 0>>` like the others, there is
 * only one meaningful value per radial grid point. All angular grid points for
 * a given radius are set to this value, namely $1-y$. Thus we only need to
 * write this value once for each radial grid point. We do this in a subfile
 * `/Cce/VolumeData/OneMinusY.dat` where the columns are named
 * `CompactifiedRadius_X` corresponding to the radial subfiles written for the
 * spin weighted quantities above (and time as the first column).
 *
 * All data will be written into the `observers::OptionTags::ReductionFileName`
 * file.
 */
class ObserveFields : public Event {
  template <typename Tag, bool IncludeSecondDeriv = true>
  // clang-format off
  using zero_one_two_radial_derivs = tmpl::flatten<tmpl::list<
      Tag,
      Tags::Dy<Tag>,
      tmpl::conditional_t<IncludeSecondDeriv,
                          Tags::Dy<Tags::Dy<Tag>>,
                          tmpl::list<>>>>;
  using spin_weighted_tags_to_observe = tmpl::flatten<
      tmpl::list<zero_one_two_radial_derivs<Tags::BondiBeta>,
                 zero_one_two_radial_derivs<Tags::BondiU>,
                 zero_one_two_radial_derivs<Tags::BondiQ>,
                 zero_one_two_radial_derivs<Tags::BondiW>,
                 zero_one_two_radial_derivs<Tags::BondiH, false>,
                 zero_one_two_radial_derivs<Tags::BondiJ>,
                 zero_one_two_radial_derivs<Tags::Du<Tags::BondiJ>>,
                 Tags::BondiR,
                 Tags::EthRDividedByR,
                 Tags::DuRDividedByR>>;
  // clang-format on

 public:
  using available_tags_to_observe =
      tmpl::push_back<spin_weighted_tags_to_observe,
                      Tags::ComplexInertialRetardedTime, Tags::OneMinusY>;

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

  using options = tmpl::list<VariablesToObserve>;

  static constexpr Options::String help =
      "Observe volume tensor fields on the characteristic grid. Writes volume "
      "quantities from the tensors listed in the 'VariablesToObserve' "
      "option to the `/Cce/VolumeData` subfile of the reduction h5 file.\n";

  ObserveFields() = default;

  ObserveFields(const std::vector<std::string>& variables_to_observe,
                const Options::Context& context = {});

  using compute_tags_for_observation_box = tmpl::list<>;

  using return_tags = tmpl::list<>;
  using argument_tags = tmpl::list<::Tags::DataBox>;

  template <typename DbTags, typename Metavariables, typename ArrayIndex,
            typename ParallelComponent>
  void operator()(const db::DataBox<DbTags>& box,
                  Parallel::GlobalCache<Metavariables>& cache,
                  const ArrayIndex& /*array_index*/,
                  const ParallelComponent* const /*component*/,
                  const ObservationValue& /*observation_value*/) const {
    // Number of points
    const size_t l_max = db::get<Tags::LMax>(box);
    const size_t l_max_plus_one_squared = square(l_max + 1);
    const size_t number_of_angular_points =
        Spectral::Swsh::number_of_swsh_collocation_points(l_max);
    const size_t number_of_radial_grid_points =
        db::get<Tags::NumberOfRadialPoints>(box);

    // Buffers/views
    std::vector<double> data_to_write(2 * l_max_plus_one_squared + 1);
    ComplexModalVector goldberg_mode_buffer{l_max_plus_one_squared};
    ComplexDataVector spin_weighted_data_view{};

    // Legend
    std::vector<std::string> file_legend;
    file_legend.reserve(2 * l_max_plus_one_squared + 1);
    file_legend.emplace_back("time");
    for (int i = 0; i <= static_cast<int>(l_max); ++i) {
      for (int j = -i; j <= i; ++j) {
        file_legend.push_back(MakeString{} << "Real Y_" << i << "," << j);
        file_legend.push_back(MakeString{} << "Imag Y_" << i << "," << j);
      }
    }

    // Time
    const double time = db::get<::Tags::Time>(box);

    // Observer writer
    auto observer_proxy = Parallel::get_parallel_component<
        ::observers::ObserverWriter<Metavariables>>(cache)[0];

    // Actual work to transform nodal data to modal data. Places result in
    // data_to_write (but starts placing data in the 1st, not 0th, element
    // because the 0th element is time). Also makes use of the
    // spin_weighted_data_view and goldberg_mode_buffer
    const auto transform_to_modal =
        [&spin_weighted_data_view, &goldberg_mode_buffer, &box, &l_max,
         &number_of_angular_points, &l_max_plus_one_squared,
         &data_to_write](auto tag_v, const auto& spin_weighted_transform,
                         auto& goldberg_modes, const size_t radial_index) {
          using tag = std::decay_t<decltype(tag_v)>;

          // Get ComplexDataVector out of SpinWeighted out of databox tag
          const ComplexDataVector& tensor = get(db::get<tag>(box)).data();

          // Make non-owning ComplexDataVector to angular data corresponding to
          // this radial index
          // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
          spin_weighted_data_view.set_data_ref(
              const_cast<ComplexDataVector&>(tensor).data() +
                  radial_index * number_of_angular_points,
              number_of_angular_points);

          // swsh_transform requires a SpinWeighted<ComplexDataVector>. It's
          // easier to make a const-view from a ComplexDataVector that is
          // already the proper size (spin_weighted_data_view) than it is to try
          // and do all the indexing into the big block of memory here in this
          // call. That is why we have spin_weighted_data_view above.
          make_const_view(make_not_null(&spin_weighted_transform.data()),
                          spin_weighted_data_view, 0,
                          spin_weighted_data_view.size());

          // libsharp_to_goldberg_modes expects
          // SpinWeighted<ComplexModalVector>, but we don't know the spin until
          // we loop over tensors, so we have goldberg_mode_buffer (a
          // ComplexModalVector) allocated properly above and just point into it
          // here.
          goldberg_modes.set_data_ref(make_not_null(&goldberg_mode_buffer));

          // Transform nodal data to modal data
          Spectral::Swsh::libsharp_to_goldberg_modes(
              make_not_null(&goldberg_modes),
              Spectral::Swsh::swsh_transform(l_max, 1, spin_weighted_transform),
              l_max);

          // Copy data into std::vector for writing (remember 0th component is
          // time and was written above).
          for (size_t i = 0; i < l_max_plus_one_squared; ++i) {
            data_to_write[2 * i + 1] = real(goldberg_modes.data()[i]);
            data_to_write[2 * i + 2] = imag(goldberg_modes.data()[i]);
          }
        };

    // The inertial retarded time is special because it's stored as a
    // Scalar<DataVector> because it's only real and only has one set of angular
    // points worth of data to write. However, all the machinery above is for a
    // SpinWeighted<ComplexDataVector>. Luckily there is a
    // ComplexInertialRetardedTime where the real part is the
    // InertialRetardedTime and the imaginary part is 0, so we use that instead
    // swapping the names and legend where necessary
    const std::string inertial_retarded_time_name =
        detail::name<Tags::ComplexInertialRetardedTime>();
    if (variables_to_observe_.count(inertial_retarded_time_name) == 1) {
      const std::string subfile_name =
          "/Cce/VolumeData/" + inertial_retarded_time_name;

      // Legend
      std::vector<std::string> inertial_retarded_time_legend;
      inertial_retarded_time_legend.reserve(l_max_plus_one_squared + 1);
      inertial_retarded_time_legend.emplace_back("time");
      for (int i = 0; i <= static_cast<int>(l_max); ++i) {
        for (int j = -i; j <= i; ++j) {
          inertial_retarded_time_legend.push_back(MakeString{} << "Y_" << i
                                                               << "," << j);
        }
      }

      // These have to be here because of the spin template
      const SpinWeighted<ComplexDataVector, 0> spin_weighted_transform{};
      SpinWeighted<ComplexModalVector, 0> goldberg_modes{};

      // Actually transform the time to complex modal data. Radial index 0
      // because this isn't volume data. It only holds one shell of data.
      transform_to_modal(Tags::ComplexInertialRetardedTime{},
                         spin_weighted_transform, goldberg_modes, 0);

      // Buffer to write
      std::vector<double> inertial_retarded_time_to_write(
          l_max_plus_one_squared + 1);

      inertial_retarded_time_to_write[0] = time;
      // Only copy real data
      for (size_t i = 0; i < l_max_plus_one_squared; ++i) {
        inertial_retarded_time_to_write[i + 1] = data_to_write[2 * i + 1];
      }

      // Send to observer writer
      Parallel::threaded_action<
          observers::ThreadedActions::WriteReductionDataRow>(
          observer_proxy, subfile_name, inertial_retarded_time_legend,
          std::make_tuple(std::move(inertial_retarded_time_to_write)));
    }

    // One minus y is also special because every angular grid point for a given
    // radius holds the same value. Thus we only need to write one double per
    // radial grid point corresponding to 1 - y. The subfile name is just the
    // name of the tag, and the column names correspond to the names of the
    // radial subfiles for the spin weighted quantities
    const std::string one_minus_y_name = detail::name<Tags::OneMinusY>();
    if (variables_to_observe_.count(one_minus_y_name) == 1) {
      const std::string subfile_name = "/Cce/VolumeData/" + one_minus_y_name;
      std::vector<double> one_minus_y_to_write;
      std::vector<std::string> one_minus_y_legend;
      one_minus_y_to_write.reserve(number_of_radial_grid_points + 1);
      one_minus_y_legend.reserve(number_of_radial_grid_points + 1);
      one_minus_y_to_write.emplace_back(time);
      one_minus_y_legend.emplace_back("time");

      const ComplexDataVector& one_minus_y =
          get(db::get<Tags::OneMinusY>(box)).data();

      // All nodal data for each radius are the same value so we just take the
      // first one
      for (size_t radial_index = 0; radial_index < number_of_radial_grid_points;
           radial_index++) {
        one_minus_y_to_write.emplace_back(
            real(one_minus_y[radial_index * number_of_angular_points]));
        one_minus_y_legend.emplace_back("CompactifiedRadius_" +
                                        std::to_string(radial_index));
      }

      // Send to observer writer
      Parallel::threaded_action<
          observers::ThreadedActions::WriteReductionDataRow>(
          observer_proxy, subfile_name, one_minus_y_legend,
          std::make_tuple(std::move(one_minus_y_to_write)));
    }

    // Everything needs the same time so we just write it once here. We use the
    // code time because the inertial retarded time is specified over the whole
    // sphere and is written above (as a function of the code time as well)
    data_to_write[0] = time;

    // Loop over all available spin weighted tags and check if we are observing
    // this tag. We just capture everything in the scope because we need a
    // majority of the variables anyways
    tmpl::for_each<spin_weighted_tags_to_observe>([&](auto tag_v) {
      using tag = tmpl::type_from<decltype(tag_v)>;
      constexpr int spin = tag::type::type::spin;
      const std::string name = detail::name<tag>();

      // If we aren't observing this tag, then skip it
      if (variables_to_observe_.count(name) != 1) {
        return;
      }

      // These have to be here because of the spin template
      const SpinWeighted<ComplexDataVector, spin> spin_weighted_transform{};
      SpinWeighted<ComplexModalVector, spin> goldberg_modes{};

      // If we are observing this tag, loop over all radii and write data to
      // separate subfiles for each radius
      for (size_t radial_index = 0; radial_index < number_of_radial_grid_points;
           radial_index++) {
        const std::string subfile_name = "/Cce/VolumeData/" + name +
                                         "/CompactifiedRadius_" +
                                         std::to_string(radial_index);

        // Actually transform the time to complex modal data.
        transform_to_modal(tag{}, spin_weighted_transform, goldberg_modes,
                           radial_index);

        // Send to observer writer
        Parallel::threaded_action<
            observers::ThreadedActions::WriteReductionDataRow>(
            observer_proxy, subfile_name, file_legend,
            std::make_tuple(data_to_write));
      }
    });
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
    p | variables_to_observe_;
  }

 private:
  std::unordered_set<std::string> variables_to_observe_{};
};

ObserveFields::ObserveFields(
    const std::vector<std::string>& variables_to_observe,
    const Options::Context& context)
    : variables_to_observe_([&context, &variables_to_observe]() {
        std::unordered_set<std::string> result{};
        for (const auto& tensor : variables_to_observe) {
          if (result.count(tensor) != 0) {
            PARSE_ERROR(
                context,
                "Listed variable '"
                    << tensor
                    << "' more than once in list of variables to observe.");
          }
          result.insert(tensor);
        }
        return result;
      }()) {
  std::unordered_set<std::string> valid_tensors{};
  tmpl::for_each<available_tags_to_observe>([&valid_tensors](auto tag_v) {
    using tag = tmpl::type_from<decltype(tag_v)>;
    valid_tensors.insert(detail::name<tag>());
  });

  for (const auto& name : variables_to_observe_) {
    if (valid_tensors.count(name) != 1) {
      PARSE_ERROR(
          context,
          name << " is not an available variable. Available variables:\n"
               << valid_tensors);
    }
  }
}

/// \cond
PUP::able::PUP_ID ObserveFields::my_PUP_ID = 0;  // NOLINT
/// \endcond
}  // namespace Events
}  // namespace Cce
