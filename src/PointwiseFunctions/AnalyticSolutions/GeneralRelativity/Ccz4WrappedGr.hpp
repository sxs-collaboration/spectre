// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
class DataVector;
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace Ccz4::Solutions {
/*!
 * \brief A wrapper for general-relativity analytic solutions that loads
 * the analytic solution and then adds a function that returns
 * any combination of the Ccz4 evolution variables.
 * Specifically, see `Ccz4::fd::System`.
 */
template <typename SolutionType>
class Ccz4WrappedGr : public virtual evolution::initial_data::InitialData,
                      public SolutionType {
 public:
  using SolutionType::SolutionType;

  Ccz4WrappedGr() = default;
  Ccz4WrappedGr(const Ccz4WrappedGr& /*rhs*/) = default;
  Ccz4WrappedGr& operator=(const Ccz4WrappedGr& /*rhs*/) = default;
  Ccz4WrappedGr(Ccz4WrappedGr&& /*rhs*/) = default;
  Ccz4WrappedGr& operator=(Ccz4WrappedGr&& /*rhs*/) = default;
  ~Ccz4WrappedGr() override = default;

  explicit Ccz4WrappedGr(const SolutionType& wrapped_solution);

  auto get_clone() const
      -> std::unique_ptr<evolution::initial_data::InitialData> override;

  /// \cond
  explicit Ccz4WrappedGr(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Ccz4WrappedGr);
  /// \endcond

  static constexpr size_t volume_dim = SolutionType::volume_dim;
  static_assert(volume_dim == 3,
                "Ccz4 evolution system has only been implemented in 3D!");
  using options = typename SolutionType::options;
  static constexpr Options::String help = SolutionType::help;
  static std::string name() {
    return "Ccz4(" + pretty_type::name<SolutionType>() + ")";
  }

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
  using tags =
      tmpl::push_back<typename SolutionType::template tags<DataType>,
                      Ccz4::Tags::ConformalMetric<DataType, volume_dim>,
                      Ccz4::Tags::ConformalFactor<DataType>,
                      Ccz4::Tags::ATilde<DataType, volume_dim>,
                      gr::Tags::TraceExtrinsicCurvature<DataType>,
                      Ccz4::Tags::Theta<DataType>,
                      Ccz4::Tags::GammaHat<DataType, volume_dim>>;

  template <typename... Tags>
  tuples::TaggedTuple<Tags...> variables(
      const tnsr::I<DataVector, volume_dim>& x, const double t,
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
                                     const double t,
                                     tmpl::list<Tag> /*meta*/) const {
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
  // Ccz4::Tags::ConformalMetric, Ccz4::Tags::ConformalFactor,
  // Ccz4:Tags::ATilde, gr::Tags::TraceExtrinsicCurvature,
  // Ccz4::Tags::Theta, Ccz4::Tags::GammaHat)
  using TagShift = gr::Tags::Shift<DataVector, volume_dim>;
  using TagSpatialMetric = gr::Tags::SpatialMetric<DataVector, volume_dim>;
  using TagInverseSpatialMetric =
      gr::Tags::InverseSpatialMetric<DataVector, volume_dim>;
  using TagExCurvature = gr::Tags::ExtrinsicCurvature<DataVector, volume_dim>;

  template <typename Tag>
  tuples::TaggedTuple<Tag> variables(
      const tnsr::I<DataVector, volume_dim>& /*x*/, double /*t*/,
      tmpl::list<Tag> /*meta*/,
      const IntermediateVars& intermediate_vars) const {
    static_assert(
        tmpl::list_contains_v<typename SolutionType::template tags<DataVector>,
                              Tag>);
    return {get<Tag>(intermediate_vars)};
  }

  template <typename Tag>
  tuples::TaggedTuple<Tag> variables(
      const tnsr::I<DataVector, volume_dim>& /*x*/, tmpl::list<Tag> /*meta*/,
      const IntermediateVars& intermediate_vars) const {
    static_assert(
        tmpl::list_contains_v<typename SolutionType::template tags<DataVector>,
                              Tag>);
    return {get<Tag>(intermediate_vars)};
  }

  tuples::TaggedTuple<Ccz4::Tags::ConformalMetric<DataVector, volume_dim>>
  variables(
      const tnsr::I<DataVector, volume_dim>& /*x*/,
      tmpl::list<Ccz4::Tags::ConformalMetric<DataVector, volume_dim>> /*meta*/,
      const IntermediateVars& intermediate_vars) const;
  tuples::TaggedTuple<Ccz4::Tags::ConformalFactor<DataVector>> variables(
      const tnsr::I<DataVector, volume_dim>& /*x*/,
      tmpl::list<Ccz4::Tags::ConformalFactor<DataVector>> /*meta*/,
      const IntermediateVars& intermediate_vars) const;
  tuples::TaggedTuple<Ccz4::Tags::ATilde<DataVector, volume_dim>> variables(
      const tnsr::I<DataVector, volume_dim>& /*x*/,
      tmpl::list<Ccz4::Tags::ATilde<DataVector, volume_dim>> /*meta*/,
      const IntermediateVars& intermediate_vars) const;
  tuples::TaggedTuple<gr::Tags::TraceExtrinsicCurvature<DataVector>> variables(
      const tnsr::I<DataVector, volume_dim>& /*x*/,
      tmpl::list<gr::Tags::TraceExtrinsicCurvature<DataVector>> /*meta*/,
      const IntermediateVars& intermediate_vars) const;
  tuples::TaggedTuple<Ccz4::Tags::Theta<DataVector>> variables(
      const tnsr::I<DataVector, volume_dim>& /*x*/,
      tmpl::list<Ccz4::Tags::Theta<DataVector>> /*meta*/,
      const IntermediateVars& intermediate_vars) const;
  tuples::TaggedTuple<Ccz4::Tags::GammaHat<DataVector, volume_dim>> variables(
      const tnsr::I<DataVector, volume_dim>& /*x*/,
      tmpl::list<Ccz4::Tags::GammaHat<DataVector, volume_dim>> /*meta*/,
      const IntermediateVars& intermediate_vars) const;

  tuples::TaggedTuple<Ccz4::Tags::ConformalMetric<DataVector, volume_dim>>
  variables(
      const tnsr::I<DataVector, volume_dim>& x, const double /*t*/,
      tmpl::list<Ccz4::Tags::ConformalMetric<DataVector, volume_dim>> meta,
      const IntermediateVars& intermediate_vars) const {
    return variables(x, meta, intermediate_vars);
  }
  tuples::TaggedTuple<Ccz4::Tags::ConformalFactor<DataVector>> variables(
      const tnsr::I<DataVector, volume_dim>& x, double /*t*/,
      tmpl::list<Ccz4::Tags::ConformalFactor<DataVector>> meta,
      const IntermediateVars& intermediate_vars) const {
    return variables(x, meta, intermediate_vars);
  }
  tuples::TaggedTuple<Ccz4::Tags::ATilde<DataVector, volume_dim>> variables(
      const tnsr::I<DataVector, volume_dim>& x, double /*t*/,
      tmpl::list<Ccz4::Tags::ATilde<DataVector, volume_dim>> meta,
      const IntermediateVars& intermediate_vars) const {
    return variables(x, meta, intermediate_vars);
  }
  tuples::TaggedTuple<gr::Tags::TraceExtrinsicCurvature<DataVector>> variables(
      const tnsr::I<DataVector, volume_dim>& x, const double /*t*/,
      tmpl::list<gr::Tags::TraceExtrinsicCurvature<DataVector>> meta,
      const IntermediateVars& intermediate_vars) const {
    return variables(x, meta, intermediate_vars);
  }
  tuples::TaggedTuple<Ccz4::Tags::Theta<DataVector>> variables(
      const tnsr::I<DataVector, volume_dim>& x, double /*t*/,
      tmpl::list<Ccz4::Tags::Theta<DataVector>> meta,
      const IntermediateVars& intermediate_vars) const {
    return variables(x, meta, intermediate_vars);
  }
  tuples::TaggedTuple<Ccz4::Tags::GammaHat<DataVector, volume_dim>> variables(
      const tnsr::I<DataVector, volume_dim>& x, double /*t*/,
      tmpl::list<Ccz4::Tags::GammaHat<DataVector, volume_dim>> meta,
      const IntermediateVars& intermediate_vars) const {
    return variables(x, meta, intermediate_vars);
  }
};

template <typename SolutionType>
bool operator==(const Ccz4WrappedGr<SolutionType>& lhs,
                const Ccz4WrappedGr<SolutionType>& rhs);

template <typename SolutionType>
bool operator!=(const Ccz4WrappedGr<SolutionType>& lhs,
                const Ccz4WrappedGr<SolutionType>& rhs);

template <typename SolutionType>
Ccz4WrappedGr(SolutionType solution) -> Ccz4WrappedGr<SolutionType>;
}  // namespace Ccz4::Solutions
