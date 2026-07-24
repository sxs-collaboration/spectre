// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <pup.h>

#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticData/ScalarTensor/ScalarField/AnalyticData.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarTensor::AnalyticData::ScalarField {

/*!
 * \brief Analytic profile for the scalar field composed by two superimposed
 * $\frac{A}{r}$ profiles.
 *
 * This profile is inteded to be used in binary black hole configurations, and
 * describes the superposition of two $\frac{A}{r}$ profiles, centered on the
 * two black holes. The expression of the profile then reads
 * \begin{equation}
 *   \Psi(x^i) = \frac{A_a}{|x^i - x^i_a|} + \frac{A_b}{|x^i - x^i_b|},
 * \end{equation}
 * where $\Psi$ is the scalar field, $x_{a,b}$ are the coordinate locations of
 * the centers of the two profiles and $A_{a,b}$ are the amplitudes. Note that
 * the norm for the distance is computed using the eucledian metric.
 */
template <size_t Dim>
class SuperposedInverser
    : public ScalarTensor::AnalyticData::ScalarField::AnalyticData<Dim> {
 public:
  struct AmplitudeA {
    using type = double;
    static constexpr Options::String help = "Scaled Amplitude A.";
  };
  struct AmplitudeB {
    using type = double;
    static constexpr Options::String help = "Scaled Amplitude B.";
  };
  struct LocationA {
    using type = std::array<double, Dim>;
    static constexpr Options::String help = "Location of black hole A.";
  };
  struct LocationB {
    using type = std::array<double, Dim>;
    static constexpr Options::String help = "location of black hole B.";
  };

  using options = tmpl::list<AmplitudeA, AmplitudeB, LocationA, LocationB>;
  static constexpr Options::String help{
      "Superposed 1/r intitial guesses for the scalar field in a black hole "
      "binary system."};

  using scalar_field_tags =
      typename ScalarTensor::AnalyticData::ScalarField::AnalyticData<
          Dim>::scalar_field_tags;

  SuperposedInverser() = default;
  SuperposedInverser(const SuperposedInverser&) = default;
  SuperposedInverser& operator=(const SuperposedInverser&) = default;
  SuperposedInverser(SuperposedInverser&&) = default;
  SuperposedInverser& operator=(SuperposedInverser&&) = default;
  ~SuperposedInverser() override = default;
  SuperposedInverser(double amplitude_a, double amplitude_b,
                     std::array<double, Dim> location_a,
                     std::array<double, Dim> location_b);

  /// \cond
  explicit SuperposedInverser(CkMigrateMessage* m)
      : ScalarTensor::AnalyticData::ScalarField::AnalyticData<Dim>(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(SuperposedInverser);  // NOLINT
  /// \endcond

  std::unique_ptr<ScalarTensor::AnalyticData::ScalarField::AnalyticData<Dim>>
  get_clone() const override;

  tuples::TaggedTuple<::CurvedScalarWave::Tags::Psi> variables(
      const tnsr::I<DataVector, Dim>& x,
      tmpl::list<::CurvedScalarWave::Tags::Psi> /*meta*/) const;

  tuples::TaggedTuple<::CurvedScalarWave::Tags::Phi<Dim, Frame::Inertial>>
  variables(
      const tnsr::I<DataVector, Dim>& x,
      tmpl::list<::CurvedScalarWave::Tags::Phi<Dim, Frame::Inertial>> /*meta*/)
      const;

  template <typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataVector, Dim>& x,
      tmpl::list<RequestedTags...> /*meta*/) const {
    static_assert(
        tmpl::size<tmpl::list_difference<tmpl::list<RequestedTags...>,
                                         scalar_field_tags>>::value == 0,
        "The requested tag is not supported");
    return {make_with_value<typename RequestedTags::type>(x, 0.)...};
  }

  void pup(PUP::er& p) override;

 private:
  double amplitude_a_{std::numeric_limits<double>::signaling_NaN()};
  double amplitude_b_{std::numeric_limits<double>::signaling_NaN()};
  std::array<double, Dim> location_a_{
      std::numeric_limits<double>::signaling_NaN()};
  std::array<double, Dim> location_b_{
      std::numeric_limits<double>::signaling_NaN()};

  template <size_t SpatialDim>
  friend bool operator==(const SuperposedInverser<SpatialDim>& lhs,
                         const SuperposedInverser<SpatialDim>& rhs);
};

template <size_t SpatialDim>
bool operator!=(const SuperposedInverser<SpatialDim>& lhs,
                const SuperposedInverser<SpatialDim>& rhs);
}  // namespace ScalarTensor::AnalyticData::ScalarField
