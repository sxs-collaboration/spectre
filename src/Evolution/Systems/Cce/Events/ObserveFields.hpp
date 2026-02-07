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
#include "DataStructures/DataBox/ObservationBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/Systems/Cce/LinearOperators.hpp"
#include "Evolution/Systems/Cce/NewmanPenrose.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/SwshDerivatives.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "IO/Observer/VolumeActions.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCoefficients.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCollocation.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshTransform.hpp"
#include "Options/Context.hpp"
#include "Options/ParseError.hpp"
#include "Options/String.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
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

namespace Cce::Events {
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
 * - `Cce::Tags::Psi0`
 * - `Cce::Tags::Psi1`
 * - `Cce::Tags::Psi2`
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
 * All data will be written into the `observers::OptionTags::VolumeFileName`
 * file. If CCE is run on a single core, then this will write the volume data
 * immediately (synchronously) instead of sending it to the ObserverWriter to be
 * written asynchronously.  The option `SubgroupName` controls the name of the
 * H5 group where this volume data is written. For example, if `SubgroupName` is
 * "CceVolumeData", then the volume file will contain
 * `/CceVolumeData/VolumeData.vol` for most fields; it would contain
 * `/CceVolumeData/InertialRetardedTime.vol` for the inertial retarded time; and
 * it would contain `/CceVolumeData/OneMinusY.vol` for the compactified radial
 * coordinate. The structure of the .vol subfiles is the same as that for DG
 * volume data. However, the extents of the three different volume files are all
 * different, and for modal directions, they take the values l_max in both of
 * the two extent slots corresponding to angular modes. This  also means that
 * complex modal data (which has real/imag parts interleaved, as described
 * below) has twice as much data as the products of the extents suggests.
 *
 * The formats for `Cce::Tags::ComplexInertialRetardedTime` and
 * `Cce::Tags::OneMinusY` are special and are described below. Every other field
 * follows the same format. Each ObservationId contains one time slice. Within
 * an ObservationId, each field is an unraveled vector of complex modal
 * coefficients at compactified radial slices (the compactified coordinate is $y
 * = 1 - 2R/r$ where $r$ is your coordinate radius and $R$ is the coordinate
 * radius of your worldtube; it is recommended to always dump the quantity
 * `Cce::Tags::OneMinusY` so the values of the compactified coordinates are
 * available as well). The ordering of the coefficients for a field $f$ at
 * constant observation event are, for example:
 * Re f_{0,0}(1-y_0), Im f_{0,0}(1-y_0), Re f_{1,-1}(1-y_0), Im f_{1,-1}(1-y_0),
 * Re f_{1,0}(1-y_0), Im f_{1,0}(1-y_0), ...
 * Re f_{l_{max},l_{max}}(1-y_0), Im f_{l_{max},l_{max}}(1-y_0),
 * Re f_{0,0}(1-y_1), Im f_{0,0}(1-y_1), ...
 * Re f_{l_{max},l_{max}}(1-y_{max)), Im f_{l_{max},l_{max}}(1-y_{max}).
 * That is, the radial slice indexes the slowest, followed by the $\ell$ number
 * (always starting at 0, even for s≠0), followed by the azimuthal m number
 * (from $-\ell$ to $+\ell$ inclusive), followed by interleaving real and
 * imaginary parts of the complex field.
 *
 * There are two notable exceptions to this format. One is
 * `Cce::Tags::ComplexInertialRetardedTime`. The quantity we are actually
 * interested in is `Cce::Tags::InertialRetardedTime` which is real and only
 * defined once for every direction $\theta,\phi$ (meaning it does not have
 * different values at the different radial grid points). However, we use
 * `Cce::Tags::ComplexInertialRetardedTime` because it has the same data type as
 * the other tags which makes the internals of the class simpler. This quantity
 * is stored in `/<SubgroupName>/InertialRetardedTime.vol`.
 *
 * The second is `Cce::Tags::OneMinusY`. Even though this quantity is stored as
 * a `Scalar<SpinWeighted<ComplexDataVector, 0>>` like the others, there is only
 * one meaningful value per radial grid point. All angular grid points for a
 * given radius are set to this value, namely $1-y$. Thus we only need to write
 * this value once for each radial grid point. We do this in a volume subfile
 * `/<SubgroupName>/OneMinusY.vol` with the elements in the same order as the
 * radial index order for the spin weighted quantities above.
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
                 Tags::Psi0,
                 Tags::Psi1,
                 Tags::Psi2,
                 Tags::NewmanPenroseAlpha,
                 Tags::NewmanPenroseBeta,
                 Tags::NewmanPenroseGamma,
                 Tags::NewmanPenroseEpsilon,
                 // Tags::NewmanPenroseKappa,
                 // in our choice of tetrad, \kappa=0
                 Tags::NewmanPenroseTau,
                 Tags::NewmanPenroseSigma,
                 Tags::NewmanPenroseRho,
                 Tags::NewmanPenrosePi,
                 Tags::NewmanPenroseNu,
                 Tags::NewmanPenroseMu,
                 Tags::NewmanPenroseLambda,
                 Tags::EthRDividedByR,
                 Tags::DuRDividedByR>>;
  // clang-format on

 public:
  using available_tags_to_observe =
      tmpl::push_back<spin_weighted_tags_to_observe,
                      Tags::ComplexInertialRetardedTime, Tags::OneMinusY>;

  /// \cond
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ObserveFields);  // NOLINT
  /// \endcond

  /// The name of the subgroup inside the HDF5 file
  struct SubgroupName {
    using type = std::string;
    static constexpr Options::String help = {
      "The name of the subgroup inside the HDF5 file without an extension and "
      "without a preceding '/'."};
  };

  struct VariablesToObserve {
    static constexpr Options::String help = "Subset of variables to observe";
    using type = std::vector<std::string>;
    static size_t lower_bound_on_size() { return 1; }
  };

  using options = tmpl::list<SubgroupName, VariablesToObserve>;

  static constexpr Options::String help =
      "Observe volume tensor fields on the characteristic grid. Writes volume "
      "quantities from the tensors listed in the 'VariablesToObserve' "
      "option to volume subfiles in the subgroup named by the option "
      "'SubgroupName', into the volume h5 file named by the option "
      "'VolumeFileName' of Observers.\n";

  ObserveFields() = default;

  ObserveFields(const std::string& subgroup_name,
                const std::vector<std::string>& variables_to_observe,
                const Options::Context& context = {});

  using compute_tags_for_observation_box =
    tmpl::list<Tags::Psi0Compute, Tags::Psi1Compute, Tags::Psi2Compute,
               Tags::SwshDerivativeCompute<Tags::BondiJ,
                                                  Spectral::Swsh::Tags::Eth>,
               Tags::SwshDerivativeCompute<Tags::BondiW,
                                                  Spectral::Swsh::Tags::Eth>,
               Tags::NewmanPenroseAlphaCompute, Tags::NewmanPenroseBetaCompute,
               Tags::NewmanPenroseGammaCompute,
               Tags::NewmanPenroseEpsilonCompute,
               // Tags::NewmanPenroseKappaCompute,
               // in our choice of tetrad, \kappa=0
               Tags::NewmanPenroseTauCompute,
               Tags::NewmanPenroseSigmaCompute, Tags::NewmanPenroseRhoCompute,
               Tags::NewmanPenrosePiCompute, Tags::NewmanPenroseNuCompute,
               Tags::NewmanPenroseMuCompute, Tags::NewmanPenroseLambdaCompute,
               Tags::SwshDerivativeCompute<Tags::NewmanPenrosePi,
                                                 Spectral::Swsh::Tags::Eth>,
               Tags::SwshDerivativeCompute<Tags::NewmanPenrosePi,
                                                 Spectral::Swsh::Tags::Ethbar>,
               Tags::DyCompute<Tags::NewmanPenrosePi>,
               Tags::DyCompute<Tags::NewmanPenroseMu>
      >;

  using return_tags = tmpl::list<>;
  using argument_tags = tmpl::list<::Tags::ObservationBox>;

  template <typename DataBoxType, typename ComputeTagsList,
            typename Metavariables, typename ArrayIndex,
            typename ParallelComponent>
  void operator()(const ObservationBox<DataBoxType, ComputeTagsList>& box,
                  Parallel::GlobalCache<Metavariables>& cache,
                  const ArrayIndex& /*array_index*/,
                  const ParallelComponent* const /*component*/,
                  const ObservationValue& /*observation_value*/) const {
    const bool write_synchronously =
        Parallel::number_of_procs<size_t>(cache) == 1;

    // Number of points
    const size_t l_max = get<Tags::LMax>(box);
    const size_t l_max_plus_one_squared = square(l_max + 1);
    const size_t number_of_angular_points =
        Spectral::Swsh::number_of_swsh_collocation_points(l_max);
    const size_t number_of_radial_grid_points =
        get<Tags::NumberOfRadialPoints>(box);

    // Time
    const double time = get<::Tags::Time>(box);

    // Observer writer
    auto observer_proxy = Parallel::get_parallel_component<
        ::observers::ObserverWriter<Metavariables>>(cache)[0];

    ////////////////////////////////////////////////////////////
    // The inertial retarded time is special because it's stored as a
    // Scalar<DataVector> because it's only real and only has one set of angular
    // points worth of data to write. However, all the machinery is for a
    // SpinWeighted<ComplexDataVector>. Luckily there is a
    // ComplexInertialRetardedTime where the real part is the
    // InertialRetardedTime and the imaginary part is 0, so we use that instead,
    // swapping the names where necessary.
    // Put into volume subfile named /<SubgroupName>/InertialRetardedTime.vol
    const std::string inertial_retarded_time_name =
        detail::name<Tags::ComplexInertialRetardedTime>();
    if (variables_to_observe_.count(inertial_retarded_time_name) == 1) {
      const std::string subfile_name = subgroup_path_
                                       + "/" + inertial_retarded_time_name;
      const observers::ObservationId observation_id{time,
        subfile_name + ".vol"};
      const std::vector<size_t> extents_vector{{l_max, l_max}};
      const std::vector<Spectral::Basis> bases_vector{
        {Spectral::Basis::SphericalHarmonic,
         Spectral::Basis::SphericalHarmonic}};
      const std::vector<Spectral::Quadrature> quadratures_vector{
        {Spectral::Quadrature::Gauss,
         Spectral::Quadrature::Equiangular}};

      const SpinWeighted<ComplexDataVector, 0>& complex_inertial_retarded_time =
          get(get<Tags::ComplexInertialRetardedTime>(box));

      // Allocate a buffer to receive the transformed data, since
      // WriteVolumeData only understands DataVectors, not
      // ComplexDataVectors.
      DataVector goldberg_modes_interleaved_dv(2 * l_max_plus_one_squared);

      // A non-owning view of goldberg_modes_interleaved_dv,
      // with the correct spin
      SpinWeighted<ComplexModalVector, 0> goldberg_mode_view;
      goldberg_mode_view.set_data_ref(
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        reinterpret_cast<std::complex<double>*>(
          goldberg_modes_interleaved_dv.data()),
        l_max_plus_one_squared);

      Spectral::Swsh::libsharp_to_goldberg_modes(
          make_not_null(&goldberg_mode_view),
          Spectral::Swsh::swsh_transform(
              l_max, 1, complex_inertial_retarded_time),
          l_max);

      const std::vector<TensorComponent> tensor_components{
        {inertial_retarded_time_name, goldberg_modes_interleaved_dv}};

      if (write_synchronously) {
        Parallel::local_synchronous_action<
          observers::ThreadedActions::WriteVolumeData>(
              observer_proxy, cache,
              Parallel::get<observers::Tags::VolumeFileName>(cache),
              subfile_name, observation_id,
              std::vector<ElementVolumeData>{
                {inertial_retarded_time_name, tensor_components,
                 extents_vector, bases_vector,
                 quadratures_vector}});
      } else {
        // Send to observer writer
        Parallel::threaded_action<
          observers::ThreadedActions::WriteVolumeData>(
              observer_proxy,
              Parallel::get<observers::Tags::VolumeFileName>(cache),
              subfile_name, observation_id,
              std::vector<ElementVolumeData>{
                {inertial_retarded_time_name, tensor_components,
                 extents_vector, bases_vector,
                 quadratures_vector}});
      }
    }

    ////////////////////////////////////////////////////////////
    // One minus y is also special because every angular grid point for a given
    // radius holds the same value. Thus we only need to write one double per
    // radial grid point corresponding to 1 - y. Put into volume subfile named
    // /<SubgroupName>/OneMinusY.vol
    const std::string one_minus_y_name = detail::name<Tags::OneMinusY>();
    if (variables_to_observe_.count(one_minus_y_name) == 1) {
      const std::string subfile_name = subgroup_path_ + "/" + one_minus_y_name;
      const observers::ObservationId observation_id{time,
        subfile_name + ".vol"};
      const std::vector<size_t> extents_vector{number_of_radial_grid_points};
      const std::vector<Spectral::Basis> bases_vector{
        Spectral::Basis::Legendre};
      const std::vector<Spectral::Quadrature> quadratures_vector{
        Spectral::Quadrature::GaussLobatto};

      const ComplexDataVector& one_minus_y =
          get(get<Tags::OneMinusY>(box)).data();

      DataVector one_minus_y_to_write(number_of_radial_grid_points);

      for (size_t radial_index = 0; radial_index < number_of_radial_grid_points;
           radial_index++) {
        one_minus_y_to_write[radial_index] =
            real(one_minus_y[radial_index * number_of_angular_points]);
      }

      const std::vector<TensorComponent> tensor_components{
        {one_minus_y_name, one_minus_y_to_write}};

      if (write_synchronously) {
        Parallel::local_synchronous_action<
          observers::ThreadedActions::WriteVolumeData>(
              observer_proxy, cache,
              Parallel::get<observers::Tags::VolumeFileName>(cache),
              subfile_name, observation_id,
              std::vector<ElementVolumeData>{
                {one_minus_y_name, tensor_components,
                 extents_vector, bases_vector,
                 quadratures_vector}});
      } else {
        // Send to observer writer
        Parallel::threaded_action<
          observers::ThreadedActions::WriteVolumeData>(
              observer_proxy,
              Parallel::get<observers::Tags::VolumeFileName>(cache),
              subfile_name, observation_id,
              std::vector<ElementVolumeData>{
                {one_minus_y_name, tensor_components,
                 extents_vector, bases_vector,
                 quadratures_vector}});
      }
    }

    ////////////////////////////////////////////////////////////
    // Everything else gets written together into the volume subfile named
    // /<SubgroupName>/VolumeData.vol

    // Field-independent info for writing into volume data file
    const std::string subfile_name = subgroup_path_ + "/VolumeData";
    const observers::ObservationId observation_id{time,
      subfile_name + ".vol"};
    const std::vector<size_t> extents_vector{
      {number_of_radial_grid_points, l_max, l_max}};
    const std::vector<Spectral::Basis> bases_vector{
      {Spectral::Basis::Legendre,
       Spectral::Basis::SphericalHarmonic,
       Spectral::Basis::SphericalHarmonic}};
    const std::vector<Spectral::Quadrature> quadratures_vector{
      {Spectral::Quadrature::GaussLobatto,
       Spectral::Quadrature::Gauss,
       Spectral::Quadrature::Equiangular}};

    // Create tensor_components by looping over all available spin
    // weighted tags and checking if we are observing this tag.
    std::vector<TensorComponent> tensor_components;
    tmpl::for_each<spin_weighted_tags_to_observe>([&](auto tag_v) {
      using tag = tmpl::type_from<decltype(tag_v)>;
      constexpr int spin = tag::type::type::spin;
      const std::string name = detail::name<tag>();

      // If we aren't observing this tag, then skip it
      if (not variables_to_observe_.contains(name)) {
        return;
      }

      const SpinWeighted<ComplexDataVector, spin>& field =
        get(get<tag>(box));

      // Allocate a buffer to receive the transformed data, since
      // WriteVolumeData only understands DataVectors, not
      // ComplexDataVectors.
      DataVector goldberg_modes_interleaved_dv(2 *
        l_max_plus_one_squared * number_of_radial_grid_points);

      // A non-owning view of goldberg_modes_interleaved_dv,
      // with the correct spin
      SpinWeighted<ComplexModalVector, spin> goldberg_mode_view;
      goldberg_mode_view.set_data_ref(
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        reinterpret_cast<std::complex<double>*>(
          goldberg_modes_interleaved_dv.data()),
        l_max_plus_one_squared * number_of_radial_grid_points);

      Spectral::Swsh::libsharp_to_goldberg_modes(
        make_not_null(&goldberg_mode_view),
        Spectral::Swsh::swsh_transform(l_max,
          number_of_radial_grid_points, field), l_max);

      tensor_components.emplace_back(
        name, std::move(goldberg_modes_interleaved_dv));

    });

    if (write_synchronously) {
      Parallel::local_synchronous_action<
        observers::ThreadedActions::WriteVolumeData>(
        observer_proxy, cache,
        Parallel::get<observers::Tags::VolumeFileName>(cache),
        subfile_name, observation_id,
        std::vector<ElementVolumeData>{
          {"VolumeData", tensor_components,
           extents_vector, bases_vector,
           quadratures_vector}});
    } else {
      // Send to observer writer
      Parallel::threaded_action<
        observers::ThreadedActions::WriteVolumeData>(
        observer_proxy,
        Parallel::get<observers::Tags::VolumeFileName>(cache),
        subfile_name, observation_id,
        std::vector<ElementVolumeData>{
          {"VolumeData", tensor_components,
           extents_vector, bases_vector,
           quadratures_vector}});
    }
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
    p | subgroup_path_;
    p | variables_to_observe_;
  }

 private:
  std::string subgroup_path_;
  std::unordered_set<std::string> variables_to_observe_;
};

ObserveFields::ObserveFields(
    const std::string& subgroup_name,
    const std::vector<std::string>& variables_to_observe,
    const Options::Context& context)
    : subgroup_path_("/" + subgroup_name),
      variables_to_observe_([&context, &variables_to_observe]() {
        std::unordered_set<std::string> result{};
        for (const auto& tensor : variables_to_observe) {
          if (result.contains(tensor)) {
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
    if (not valid_tensors.contains(name)) {
      PARSE_ERROR(
          context,
          name << " is not an available variable. Available variables:\n"
               << valid_tensors);
    }
  }
}

/// \cond
#if defined(SPECTRE_USE_CHARM)
PUP::able::PUP_ID ObserveFields::my_PUP_ID = 0;  // NOLINT
#endif                                           // SPECTRE_USE_CHARM
/// \endcond
}  // namespace Cce::Events
