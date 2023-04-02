// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/CurvedScalarWave/Worldtube/PunctureField.hpp"

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/DynamicBuffer.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Utilities/Gsl.hpp"

namespace CurvedScalarWave::Worldtube {

void puncture_field_0(
    const gsl::not_null<Variables<tmpl::list<
        CurvedScalarWave::Tags::Psi, ::Tags::dt<CurvedScalarWave::Tags::Psi>,
        ::Tags::deriv<CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                      Frame::Inertial>>>*>
        result,
    const tnsr::I<DataVector, 3, Frame::Inertial>& coords, const double time,
    const double orbital_radius, const double BH_mass) {
  const size_t grid_size = get<0>(coords).size();
  result->initialize(grid_size);
  const double r0 = orbital_radius;
  const double M = BH_mass;
  const double w = 1. / (r0 * sqrt(r0));
  const double t = time;

  const double charge_pos_x = r0 * cos(w * time);
  const double charge_pos_y = r0 * sin(w * time);
  const double charge_pos_z = 0.;

  const DataVector& x = get<0>(coords);
  const DataVector& y = get<1>(coords);
  const DataVector& z = get<2>(coords);

  const DataVector Dx = x - charge_pos_x;
  const DataVector Dy = y - charge_pos_y;
  const DataVector Dz = z - charge_pos_z;

  // we use a dynamic buffer even though the size is known at compile
  // time because TempBuffer only accepts 256 arguments and takes much
  // longer to compile. The performance loss was measured to be about 10%.
  DynamicBuffer<DataVector> temps(66, grid_size);

  const double d_0 = 1.0 / r0;
  const double d_1 = 3.0 * M;
  const double d_2 = d_1 - r0;
  const double d_3 = M * sqrt(M);
  const double d_4 = r0 * r0;
  const double d_5 = sqrt(-d_2 * d_4);
  const double d_6 = d_3 * d_5;
  const double d_7 = d_6 * sqrt(-d_0 * d_2);
  const double d_8 = 4.0 * d_7;
  const double d_9 = t * w;
  const double d_10 = cos(d_9);
  const double d_11 = sin(d_9);
  const double d_12 = d_10 * d_11;
  const double d_13 = d_11 * d_11;
  const double d_14 = d_10 * d_10;
  const double d_15 = d_4 * d_4 * sqrt(d_4);
  const double d_16 = r0 * r0 * r0;
  const double d_17 = M * M;
  const double d_18 = r0 * r0 * r0 * r0;
  const double d_19 = M * M * M;
  const double d_20 = 6.0 * d_19;
  const double d_21 = 1.0 / d_4;
  const double d_22 = d_15 * d_21;
  const double d_23 = 6.0 * M;
  const double d_24 = 9.0 * d_17;
  const double d_25 = d_22 - d_23 * d_4 + d_24 * r0;
  const double d_26 = 1.0 / d_25;
  const double d_27 = d_21 * d_26;
  const double d_28 = -d_2;
  const double d_29 = sqrt(d_28 * d_4);
  const double d_30 = d_29 * d_3 * r0 * sqrt(d_0 * d_28);
  const double d_31 = 4.0 * d_30;
  const double d_32 = 1.0 / (d_2 * d_2);
  const double d_33 = d_32 * r0 * 1.0 / d_22;
  const double d_34 = 1.0 / d_15;
  const double d_35 = d_16 * d_34;
  const double d_36 = 1.0 / d_29;
  const double d_37 = sqrt(M) * r0;
  const double d_38 = 2.0 * M;
  const double d_39 = sqrt(d_4);
  const double d_40 = 1.0 / d_39;
  const double d_41 = sqrt(-d_1 * d_40 + 1);
  const double d_42 = 1.0 / d_41;
  const double d_43 = d_0 * d_42;
  const double d_44 = d_21 * d_42;
  const double d_45 = 2.0 * d_44;
  const double d_46 = M * d_4;
  const double d_47 = 1.0 / d_5;
  const double d_48 = d_38 * d_40;
  const double d_49 = d_14 * d_48 + 1;
  const double d_50 = M * d_13;
  const double d_51 = d_41 * d_6;
  const double d_52 = d_17 * r0;
  const double d_53 = 2.0 * d_7;
  const double d_54 = d_23 * r0;
  const double d_55 = d_39 * r0;
  const double d_56 = 1.0 / (d_24 - d_54 + d_55);
  const double d_57 = d_13 * d_46;
  const double d_58 = d_14 * d_46;
  const double d_59 = d_13 * d_52;
  const double d_60 = d_14 * d_52;
  const double d_61 = d_12 * d_7;
  const double d_62 = d_10 * d_10 * d_10;
  const double d_63 = d_11 * d_11 * d_11;
  const double d_64 = d_13 + d_14;
  const double d_65 = d_38 * d_43;
  const double d_66 = d_17 * d_4;
  const double d_67 = d_22 * d_50;
  const double d_68 = M * d_14;
  const double d_69 = d_22 * d_68;
  const double d_70 = d_13 * d_66;
  const double d_71 = d_14 * d_66;
  const double d_72 = d_12 * d_30;
  const double d_73 = d_36 * d_37;
  const double d_74 = 2.0 * d_51;
  const double d_75 = d_16 * d_17;
  const double d_76 = -d_10 * d_65 + d_11 * d_37 * d_47;
  const double d_77 = 2.0 * d_30;
  const double d_78 = d_10 * d_73 + d_11 * d_65;
  const double d_79 = d_13 * d_24;
  const double d_80 = d_14 * d_24;
  const double d_81 = d_18 * d_23;
  DataVector& dv_0 = temps.at(0);
  dv_0 = Dx * Dx;
  DataVector& dv_1 = temps.at(1);
  dv_1 = Dy * Dy;
  DataVector& dv_2 = temps.at(2);
  dv_2 = -dv_1;
  DataVector& dv_3 = temps.at(3);
  dv_3 = dv_0 + dv_2;
  DataVector& dv_4 = temps.at(4);
  dv_4 = Dy * d_13;
  DataVector& dv_5 = temps.at(5);
  dv_5 = Dx * dv_4;
  DataVector& dv_6 = temps.at(6);
  dv_6 = Dx * d_14;
  DataVector& dv_7 = temps.at(7);
  dv_7 = Dy * dv_6;
  DataVector& dv_8 = temps.at(8);
  dv_8 = dv_5 - dv_7;
  DataVector& dv_9 = temps.at(9);
  dv_9 = d_4 * (-d_12 * dv_3 - dv_8);
  DataVector& dv_10 = temps.at(10);
  dv_10 = Dz * Dz;
  DataVector& dv_11 = temps.at(11);
  dv_11 = dv_1 + dv_10;
  DataVector& dv_12 = temps.at(12);
  dv_12 = dv_0 + dv_11;
  DataVector& dv_13 = temps.at(13);
  dv_13 = Dx * d_10;
  DataVector& dv_14 = temps.at(14);
  dv_14 = Dy * d_11;
  DataVector& dv_15 = temps.at(15);
  dv_15 = dv_13 * dv_14;
  DataVector& dv_16 = temps.at(16);
  dv_16 = 10.0 * dv_15;
  DataVector& dv_17 = temps.at(17);
  dv_17 = 6.0 * dv_1;
  DataVector& dv_18 = temps.at(18);
  dv_18 = 9.0 * dv_10;
  DataVector& dv_19 = temps.at(19);
  dv_19 = 6.0 * dv_0;
  DataVector& dv_20 = temps.at(20);
  dv_20 = d_17 * (d_13 * (dv_1 + dv_18 + dv_19) +
                  d_14 * (dv_0 + dv_17 + dv_18) - dv_16);
  DataVector& dv_21 = temps.at(21);
  dv_21 = 2.0 * dv_15;
  DataVector& dv_22 = temps.at(22);
  dv_22 = 4.0 * dv_0;
  DataVector& dv_23 = temps.at(23);
  dv_23 = 5.0 * dv_1;
  DataVector& dv_24 = temps.at(24);
  dv_24 = 6.0 * dv_10;
  DataVector& dv_25 = temps.at(25);
  dv_25 = 5.0 * dv_0;
  DataVector& dv_26 = temps.at(26);
  dv_26 = 4.0 * dv_1;
  DataVector& dv_27 = temps.at(27);
  dv_27 = M * (d_13 * (dv_24 + dv_25 + dv_26) + d_14 * (dv_22 + dv_23 + dv_24) -
               dv_21);
  DataVector& dv_28 = temps.at(28);
  dv_28 = dv_13 + dv_14;
  DataVector& dv_29 = temps.at(29);
  dv_29 = d_20 * (dv_28 * dv_28);
  DataVector& dv_30 = temps.at(30);
  dv_30 = d_15 * dv_12 + d_16 * dv_20 - d_18 * dv_27 + d_4 * dv_29;
  DataVector& dv_31 = temps.at(31);
  dv_31 = 2.0 * dv_10;
  DataVector& dv_32 = temps.at(32);
  dv_32 = -dv_0;
  DataVector& dv_33 = temps.at(33);
  dv_33 = 2.0 * dv_1;
  DataVector& dv_34 = temps.at(31);
  dv_34 = d_13 * (-2 * dv_0 - dv_2 - dv_31) + d_14 * (-dv_31 - dv_32 - dv_33) +
          6.0 * dv_15;
  DataVector& dv_35 = temps.at(2);
  dv_35 = d_35 * dv_28;
  DataVector& dv_36 = temps.at(34);
  dv_36 = Dx * d_11;
  DataVector& dv_37 = temps.at(35);
  dv_37 = Dy * d_10;
  DataVector& dv_38 = temps.at(36);
  dv_38 = -dv_37;
  DataVector& dv_39 = temps.at(37);
  dv_39 = dv_36 + dv_38;
  DataVector& dv_40 = temps.at(38);
  dv_40 = d_37 * dv_39;
  DataVector& dv_41 = temps.at(39);
  dv_41 = d_38 * dv_28;
  DataVector& dv_42 = temps.at(40);
  dv_42 = -d_43 * dv_41;
  DataVector& dv_43 = temps.at(41);
  dv_43 = d_36 * dv_40 + dv_42;
  DataVector& dv_44 = temps.at(42);
  dv_44 = 4.0 * dv_15;
  DataVector& dv_45 = temps.at(43);
  dv_45 = d_13 * (dv_10 + dv_3);
  DataVector& dv_46 = temps.at(32);
  dv_46 = dv_11 + dv_32;
  DataVector& dv_47 = temps.at(44);
  dv_47 = -d_45 * dv_43 * (-d_14 * dv_46 + dv_44 - dv_45) + dv_34 * dv_35;
  DataVector& dv_48 = temps.at(45);
  dv_48 = d_46 * dv_47;
  DataVector& dv_49 = temps.at(38);
  dv_49 = d_47 * dv_40 + dv_42;
  DataVector& dv_50 = temps.at(33);
  dv_50 = M * d_40 * dv_44 + d_40 * d_50 * dv_33 + d_49 * dv_0 + dv_11 +
          dv_49 * dv_49;
  DataVector& dv_51 = temps.at(11);
  dv_51 = 1.0 / dv_50;
  DataVector& dv_52 = temps.at(40);
  dv_52 = d_25 * dv_51;
  DataVector& dv_53 = temps.at(46);
  dv_53 = dv_48 * dv_52;
  DataVector& dv_54 = temps.at(47);
  dv_54 = 4.0 * d_51 * dv_9 + dv_30;
  DataVector& dv_55 = temps.at(48);
  dv_55 = 1.0 / dv_54;
  DataVector& dv_56 = temps.at(49);
  dv_56 = (1.0 / 2.0) * dv_53 * dv_55;
  DataVector& dv_57 = temps.at(28);
  dv_57 = d_20 * dv_28;
  DataVector& dv_58 = temps.at(8);
  dv_58 = d_12 * dv_0 - d_12 * dv_1 + dv_8;
  DataVector& dv_59 = temps.at(50);
  dv_59 = d_14 * d_19 * dv_19;
  DataVector& dv_60 = temps.at(51);
  dv_60 = d_13 * d_19 * dv_17;
  DataVector& dv_61 = temps.at(52);
  dv_61 = d_14 * dv_0;
  DataVector& dv_62 = temps.at(53);
  dv_62 = d_13 * dv_1;
  DataVector& dv_63 = temps.at(54);
  dv_63 = d_46 * dv_24;
  DataVector& dv_64 = temps.at(55);
  dv_64 = 12.0 * d_19 * dv_15;
  DataVector& dv_manual = get(get<CurvedScalarWave::Tags::Psi>(*result));
  dv_manual = d_0 * d_56 *
              (-d_13 * dv_63 - d_14 * dv_63 + d_22 * dv_0 + d_22 * dv_1 +
               d_22 * dv_10 + d_46 * dv_21 - d_52 * dv_16 + d_52 * dv_61 +
               d_52 * dv_62 - d_57 * dv_25 - d_57 * dv_26 - d_58 * dv_22 -
               d_58 * dv_23 + d_59 * dv_18 + d_59 * dv_19 + d_60 * dv_17 +
               d_60 * dv_18 - d_61 * dv_22 + d_61 * dv_26 - d_8 * dv_5 +
               d_8 * dv_7 + dv_59 + dv_60 + dv_64);
  DataVector& dv_65 = temps.at(54);
  dv_65 = 1. / dv_manual / sqrt(dv_manual);
  DataVector& dv_66 = temps.at(21);
  dv_66 = d_26 * dv_65;
  DataVector& dv_67 = temps.at(56);
  dv_67 = dv_36 + dv_37;
  DataVector& dv_68 = temps.at(57);
  dv_68 =
      Dx * d_62 + Dy * d_63 + 2.0 * d_12 * dv_67 - d_13 * dv_13 - d_14 * dv_14;
  DataVector& dv_69 = temps.at(58);
  dv_69 = d_4 * dv_68;
  DataVector& dv_70 = temps.at(59);
  dv_70 = 5.0 * dv_37;
  DataVector& dv_71 = temps.at(60);
  dv_71 = 5.0 * dv_36;
  DataVector& dv_72 = temps.at(61);
  dv_72 = d_14 * dv_36;
  DataVector& dv_73 = temps.at(62);
  dv_73 = d_13 * dv_37;
  DataVector& dv_74 = temps.at(63);
  dv_74 = dv_72 - dv_73;
  DataVector& dv_75 = temps.at(64);
  dv_75 =
      M * (d_13 * (-4 * dv_37 + dv_71) + d_14 * (4 * dv_36 - dv_70) + dv_74);
  DataVector& dv_76 = temps.at(59);
  dv_76 = -d_13 * dv_70 + d_13 * (6 * dv_36 + dv_38) + d_14 * dv_71 +
          d_14 * (dv_36 - 6.0 * dv_37);
  DataVector& dv_77 = temps.at(60);
  dv_77 = -d_15 * dv_39 - d_16 * d_17 * dv_76 + d_18 * dv_75;
  DataVector& dv_78 = temps.at(32);
  dv_78 = -d_14 * dv_46 + dv_44 - dv_45;
  DataVector& dv_79 = temps.at(24);
  dv_79 = d_22 * dv_24;
  DataVector& dv_80 = temps.at(19);
  dv_80 = sqrt(d_33 *
               (d_18 * dv_0 + d_18 * dv_1 + d_18 * dv_10 + d_22 * d_38 * dv_15 -
                d_31 * dv_5 + d_31 * dv_7 - d_50 * dv_79 - d_66 * dv_16 +
                d_66 * dv_61 + d_66 * dv_62 - d_67 * dv_25 - d_67 * dv_26 -
                d_68 * dv_79 - d_69 * dv_22 - d_69 * dv_23 + d_70 * dv_18 +
                d_70 * dv_19 + d_71 * dv_17 + d_71 * dv_18 - d_72 * dv_22 +
                d_72 * dv_26 + dv_59 * r0 + dv_60 * r0 + dv_64 * r0));
  DataVector& dv_81 = temps.at(26);
  dv_81 = d_25 * dv_55 * dv_80;
  DataVector& dv_82 = temps.at(50);
  dv_82 = dv_51 * dv_81;
  DataVector& dv_83 = temps.at(33);
  dv_83 = 1.0 / (dv_50 * dv_50);
  DataVector& dv_84 = temps.at(47);
  dv_84 = dv_80 * 1.0 / (dv_54 * dv_54);
  DataVector& dv_85 = temps.at(25);
  dv_85 = 1.0 / dv_80;
  DataVector& dv_86 = temps.at(22);
  dv_86 = Dy * d_14;
  DataVector& dv_87 = temps.at(13);
  dv_87 = d_11 * dv_13;
  DataVector& dv_88 = temps.at(24);
  dv_88 = dv_4 - dv_86 + 2.0 * dv_87;
  DataVector& dv_89 = temps.at(51);
  dv_89 = d_4 * dv_88;
  DataVector& dv_90 = temps.at(15);
  dv_90 = Dx * d_13;
  DataVector& dv_91 = temps.at(14);
  dv_91 = d_10 * dv_14;
  DataVector& dv_92 = temps.at(55);
  dv_92 = dv_6 + 6.0 * dv_90 - 5.0 * dv_91;
  DataVector& dv_93 = temps.at(10);
  dv_93 = M * (4 * dv_6 + 5.0 * dv_90 - dv_91);
  DataVector& dv_94 = temps.at(23);
  dv_94 = d_10 * dv_57;
  DataVector& dv_95 = temps.at(16);
  dv_95 = Dx * d_15 - d_18 * dv_93 + d_4 * dv_94 + d_75 * dv_92;
  DataVector& dv_96 = temps.at(53);
  dv_96 = d_27 * dv_65;
  DataVector& dv_97 = temps.at(17);
  dv_97 = 2.0 * dv_35;
  DataVector& dv_98 = temps.at(31);
  dv_98 = d_35 * dv_34;
  DataVector& dv_99 = temps.at(18);
  dv_99 = dv_6 - dv_90 + 2.0 * dv_91;
  DataVector& dv_100 = temps.at(1);
  dv_100 = d_46 * dv_82;
  DataVector& dv_101 = temps.at(0);
  dv_101 = (1.0 / 2.0) * dv_100;
  DataVector& dv_102 = temps.at(45);
  dv_102 = dv_48 * dv_81 * dv_83;
  DataVector& dv_103 = temps.at(46);
  dv_103 = dv_53 * dv_84;
  DataVector& dv_104 = temps.at(26);
  dv_104 = d_4 * dv_99;
  DataVector& dv_105 = temps.at(52);
  dv_105 = dv_4 + 6.0 * dv_86 - 5.0 * dv_87;
  DataVector& dv_106 = temps.at(43);
  dv_106 = M * (4 * dv_4 + 5.0 * dv_86 - dv_87);
  DataVector& dv_107 = temps.at(36);
  dv_107 = d_11 * dv_57;
  DataVector& dv_108 = temps.at(65);
  dv_108 = Dy * d_15 - d_18 * dv_106 + d_4 * dv_107 + d_75 * dv_105;

  get(get<CurvedScalarWave::Tags::Psi>(*result)) =
      dv_56 * sqrt(d_33 * (d_18 * dv_12 - d_22 * dv_27 +
                           d_31 * (-d_12 * dv_3 - dv_5 + dv_7) + d_4 * dv_20 +
                           dv_29 * r0)) +
      1. / sqrt(d_27 * (d_8 * dv_9 + dv_30));
  get(get<::Tags::dt<CurvedScalarWave::Tags::Psi>>(*result)) =
      w *
      (M * d_16 * d_25 * dv_47 * dv_55 * dv_80 * dv_83 *
           (d_48 * dv_72 - d_49 * dv_36 - d_64 * d_73 * dv_43 + dv_37) -
       M * d_16 * dv_47 * dv_52 * dv_84 * (-d_74 * dv_69 - dv_77) -
       d_0 * dv_66 * (-d_53 * dv_69 - dv_77) -
       d_32 * d_40 * dv_56 * dv_85 *
           (-d_17 * d_4 * dv_76 - d_18 * dv_39 + d_22 * dv_75 +
            d_53 * dv_68 * r0) -
       d_4 * dv_82 *
           (d_18 * d_34 * dv_41 * (Dx * d_63 - Dy * d_62 + dv_74) +
            d_3 * d_42 * d_47 * d_64 * dv_78 -
            d_65 * dv_49 *
                (d_13 * dv_67 - d_14 * dv_67 + 2.0 * dv_72 - 2.0 * dv_73)) -
       dv_66 * (-d_46 * dv_58 + 5.0 * d_52 * dv_58 +
                d_53 * (d_13 * dv_3 - d_14 * dv_3 - dv_44) - dv_39 * dv_57));
  get<0>(get<::Tags::deriv<CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                           Frame::Inertial>>(*result)) =
      (1.0 / 2.0) * M * d_25 * d_32 * d_40 * dv_47 * dv_51 * dv_55 * dv_85 *
          r0 *
          (Dx * d_18 - d_22 * dv_93 + d_66 * dv_92 - d_77 * dv_88 +
           dv_94 * r0) -
      dv_101 * (-d_10 * dv_98 + 2.0 * d_21 * d_42 * d_76 * dv_78 +
                4.0 * d_21 * d_42 * dv_49 * dv_99 -
                dv_97 * (dv_6 - 2.0 * dv_90 + 3.0 * dv_91)) -
      dv_102 * (Dx * d_49 + d_48 * dv_91 + d_76 * dv_49) -
      dv_103 * (-d_74 * dv_89 + dv_95) - dv_96 * (-d_53 * dv_89 + dv_95);
  get<1>(get<::Tags::deriv<CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                           Frame::Inertial>>(*result)) =
      (1.0 / 2.0) * M * d_25 * d_32 * d_40 * dv_47 * dv_51 * dv_55 * dv_85 *
          r0 *
          (Dy * d_18 - d_22 * dv_106 + d_66 * dv_105 + d_77 * dv_99 +
           dv_107 * r0) -
      dv_101 *
          (-d_11 * dv_98 + 4.0 * d_21 * d_42 * dv_49 * dv_88 -
           d_45 * d_78 * dv_78 - dv_97 * (dv_4 - 2.0 * dv_86 + 3.0 * dv_87)) -
      dv_102 * (Dy + d_48 * dv_4 + d_48 * dv_87 - d_78 * dv_49) -
      dv_103 * (d_74 * dv_104 + dv_108) - dv_96 * (d_53 * dv_104 + dv_108);
  get<2>(get<::Tags::deriv<CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                           Frame::Inertial>>(*result)) =
      Dz * ((1.0 / 2.0) * M * d_25 * d_32 * d_40 * dv_47 * dv_51 * dv_55 *
                dv_85 * r0 * (d_18 - d_22 * d_23 * d_64 + d_24 * d_4 * d_64) -
            d_56 * dv_65 * (-d_13 * d_54 - d_14 * d_54 + d_55 + d_79 + d_80) -
            2.0 * d_64 * dv_100 * (-d_44 * dv_49 + dv_35) - dv_102 -
            dv_103 * (-d_13 * d_81 - d_14 * d_81 + d_15 + d_16 * d_79 +
                      d_16 * d_80));
}
}  // namespace CurvedScalarWave::Worldtube
