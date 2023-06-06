// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/preprocessor/list/for_each.hpp>
#include <boost/preprocessor/tuple/to_list.hpp>
#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"  // for tags
#include "Options/String.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma: no_forward_declare Tags::deriv

/// \cond
class DataVector;
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace gh {
namespace Solutions {

/*!
 * \brief A wrapper for general-relativity analytic solutions that loads
 * the analytic solution and then adds a function that returns
 * any combination of the generalized-harmonic evolution variables,
 * specifically `gr::Tags::SpacetimeMetric`, `gh::Tags::Pi`,
 * and `gh::Tags::Phi`
 */
template <typename SolutionType>
class WrappedGr : public virtual evolution::initial_data::InitialData,
                  public SolutionType {
 public:
  using SolutionType::SolutionType;

  WrappedGr() = default;
  WrappedGr(const WrappedGr& /*rhs*/) = default;
  WrappedGr& operator=(const WrappedGr& /*rhs*/) = default;
  WrappedGr(WrappedGr&& /*rhs*/) = default;
  WrappedGr& operator=(WrappedGr&& /*rhs*/) = default;
  ~WrappedGr() override = default;

  explicit WrappedGr(const SolutionType& wrapped_solution)
      : SolutionType(wrapped_solution) {}

  auto get_clone() const
      -> std::unique_ptr<evolution::initial_data::InitialData> override;

  /// \cond
  explicit WrappedGr(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(WrappedGr);
  /// \endcond

  static constexpr size_t volume_dim = SolutionType::volume_dim;
  using options = typename SolutionType::options;
  static constexpr Options::String help = SolutionType::help;
  static std::string name() { return pretty_type::name<SolutionType>(); }

  using DerivLapse = ::Tags::deriv<gr::Tags::Lapse<DataVector>,
                                   tmpl::size_t<volume_dim>, Frame::Inertial>;
  using DerivShift = ::Tags::deriv<gr::Tags::Shift<DataVector, volume_dim>,
                                   tmpl::size_t<volume_dim>, Frame::Inertial>;
  using DerivSpatialMetric =
      ::Tags::deriv<gr::Tags::SpatialMetric<DataVector, volume_dim>,
                    tmpl::size_t<volume_dim>, Frame::Inertial>;
  using TimeDerivLapse = ::Tags::dt<gr::Tags::Lapse<DataVector>>;
  using TimeDerivShift = ::Tags::dt<gr::Tags::Shift<DataVector, volume_dim>>;
  using TimeDerivSpatialMetric =
      ::Tags::dt<gr::Tags::SpatialMetric<DataVector, volume_dim>>;

  using IntermediateVars = tuples::tagged_tuple_from_typelist<
      typename SolutionType::template tags<DataVector>>;

  template <typename DataType>
  using tags = tmpl::push_back<typename SolutionType::template tags<DataType>,
                               gr::Tags::SpacetimeMetric<DataType, volume_dim>,
                               gh::Tags::Pi<DataVector, volume_dim>,
                               gh::Tags::Phi<DataVector, volume_dim>>;

  template <typename... Tags>
  tuples::TaggedTuple<Tags...> variables(
      const tnsr::I<DataVector, volume_dim>& x, double t,
      tmpl::list<Tags...> /*meta*/) const {
    // Get the underlying solution's variables using the solution's tags list,
    // store in IntermediateVariables
    const IntermediateVars& intermediate_vars = SolutionType::variables(
        x, t, typename SolutionType::template tags<DataVector>{});

    return {
        get<Tags>(variables(x, t, tmpl::list<Tags>{}, intermediate_vars))...};
  }

  template <typename Tag>
  tuples::TaggedTuple<Tag> variables(const tnsr::I<DataVector, volume_dim>& x,
                                     double t, tmpl::list<Tag> /*meta*/) const {
    const IntermediateVars& intermediate_vars = SolutionType::variables(
        x, t, typename SolutionType::template tags<DataVector>{});
    return {get<Tag>(variables(x, t, tmpl::list<Tag>{}, intermediate_vars))};
  }

  // overloads for wrapping analytic data

  template <typename... Tags>
  tuples::TaggedTuple<Tags...> variables(
      const tnsr::I<DataVector, volume_dim>& x,
      tmpl::list<Tags...> /*meta*/) const {
    // Get the underlying solution's variables using the solution's tags list,
    // store in IntermediateVariables
    const IntermediateVars intermediate_vars = SolutionType::variables(
        x, typename SolutionType::template tags<DataVector>{});

    return {get<Tags>(variables(x, tmpl::list<Tags>{}, intermediate_vars))...};
  }

  template <typename Tag>
  tuples::TaggedTuple<Tag> variables(const tnsr::I<DataVector, volume_dim>& x,
                                     tmpl::list<Tag> /*meta*/) const {
    const IntermediateVars intermediate_vars = SolutionType::variables(
        x, typename SolutionType::template tags<DataVector>{});
    return {get<Tag>(variables(x, tmpl::list<Tag>{}, intermediate_vars))};
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

 private:
  // Preprocessor logic to avoid declaring variables() functions for
  // tags other than the three the wrapper adds (i.e., other than
  // gr::Tags::SpacetimeMetric, gh::Tags::Pi, and
  // GeneralizedHarmonic:Tags::Phi)
  using TagShift = gr::Tags::Shift<DataVector, volume_dim>;
  using TagSpatialMetric = gr::Tags::SpatialMetric<DataVector, volume_dim>;
  using TagInverseSpatialMetric =
      gr::Tags::InverseSpatialMetric<DataVector, volume_dim>;
  using TagExCurvature = gr::Tags::ExtrinsicCurvature<DataVector, volume_dim>;

  template <
      typename Tag,
      Requires<tmpl::list_contains_v<
          typename SolutionType::template tags<DataVector>, Tag>> = nullptr>
  tuples::TaggedTuple<Tag> variables(
      const tnsr::I<DataVector, volume_dim>& /*x*/, double /*t*/,
      tmpl::list<Tag> /*meta*/,
      const IntermediateVars& intermediate_vars) const {
    return {get<Tag>(intermediate_vars)};
  }

  template <
      typename Tag,
      Requires<tmpl::list_contains_v<
          typename SolutionType::template tags<DataVector>, Tag>> = nullptr>
  tuples::TaggedTuple<Tag> variables(
      const tnsr::I<DataVector, volume_dim>& /*x*/, tmpl::list<Tag> /*meta*/,
      const IntermediateVars& intermediate_vars) const {
    return {get<Tag>(intermediate_vars)};
  }

  template <
      typename Tag,
      Requires<not tmpl::list_contains_v<
          typename SolutionType::template tags<DataVector>, Tag>> = nullptr>
  tuples::TaggedTuple<Tag> variables(
      const tnsr::I<DataVector, volume_dim>& x, double /*t*/,
      tmpl::list<Tag> tag_list,
      const IntermediateVars& intermediate_vars) const {
    return variables(x, tag_list, intermediate_vars);
  }

  tuples::TaggedTuple<gr::Tags::SpacetimeMetric<DataVector, volume_dim>>
  variables(
      const tnsr::I<DataVector, volume_dim>& /*x*/,
      tmpl::list<gr::Tags::SpacetimeMetric<DataVector, volume_dim>> /*meta*/,
      const IntermediateVars& intermediate_vars) const;
  tuples::TaggedTuple<gh::Tags::Pi<DataVector, volume_dim>> variables(
      const tnsr::I<DataVector, volume_dim>& /*x*/,
      tmpl::list<gh::Tags::Pi<DataVector, volume_dim>> /*meta*/,
      const IntermediateVars& intermediate_vars) const;
  tuples::TaggedTuple<gh::Tags::Phi<DataVector, volume_dim>> variables(
      const tnsr::I<DataVector, volume_dim>& /*x*/,
      tmpl::list<gh::Tags::Phi<DataVector, volume_dim>> /*meta*/,
      const IntermediateVars& intermediate_vars) const;
};

template <typename SolutionType>
bool operator==(const WrappedGr<SolutionType>& lhs,
                const WrappedGr<SolutionType>& rhs);

template <typename SolutionType>
bool operator!=(const WrappedGr<SolutionType>& lhs,
                const WrappedGr<SolutionType>& rhs);

template <typename SolutionType>
WrappedGr(SolutionType solution) -> WrappedGr<SolutionType>;
}  // namespace Solutions
}  // namespace gh
