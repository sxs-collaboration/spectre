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

// NOLINTNEXTLINE(google-readability-function-size, readability-function-size)
void puncture_field_1(
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
  DynamicBuffer<DataVector> temps(225, grid_size);

  const double d_0 = t * w;
  const double d_1 = cos(d_0);
  const double d_2 = sin(d_0);
  const double d_3 = d_1 * d_2;
  const double d_4 = d_2 * d_2;
  const double d_5 = d_1 * d_1;
  const double d_6 = 1.0 / r0;
  const double d_7 = 3.0 * M;
  const double d_8 = d_7 - r0;
  const double d_9 = sqrt(-d_6 * d_8);
  const double d_10 = r0 * r0;
  const double d_11 = sqrt(-d_10 * d_8);
  const double d_12 = d_11 * d_9;
  const double d_13 = M * sqrt(M);
  const double d_14 = 4.0 * d_10;
  const double d_15 = d_13 * d_14;
  const double d_16 = r0 * r0 * r0 * r0;
  const double d_17 = d_10 * d_10 * sqrt(d_10);
  const double d_18 = r0 * r0 * r0;
  const double d_19 = M * M;
  const double d_20 = M * M * M;
  const double d_21 = 6.0 * d_20;
  const double d_22 = 1.0 / d_10;
  const double d_23 = d_10 * sqrt(d_10);
  const double d_24 = 6.0 * M;
  const double d_25 = 9.0 * d_19;
  const double d_26 = -d_10 * d_24 + d_23 + d_25 * r0;
  const double d_27 = 1.0 / d_26;
  const double d_28 = d_22 * d_27;
  const double d_29 = 4.0 * r0;
  const double d_30 = d_13 * d_29;
  const double d_31 = 1.0 / (d_8 * d_8);
  const double d_32 = r0 * 1.0 / d_23;
  const double d_33 = d_31 * d_32;
  const double d_34 = 1.0 / d_17;
  const double d_35 = d_18 * d_34;
  const double d_36 = sqrt(d_10);
  const double d_37 = 1.0 / d_36;
  const double d_38 = d_37 * d_7 - 1;
  const double d_39 = -d_38;
  const double d_40 = sqrt(d_39);
  const double d_41 = 1.0 / d_40;
  const double d_42 = 2.0 * M;
  const double d_43 = d_42 * d_6;
  const double d_44 = d_41 * d_43;
  const double d_45 = 1.0 / d_11;
  const double d_46 = sqrt(M);
  const double d_47 = d_46 * r0;
  const double d_48 = d_22 * d_41;
  const double d_49 = d_37 * d_42;
  const double d_50 = d_49 * d_5 + 1;
  const double d_51 = M * d_37;
  const double d_52 = M * d_10;
  const double d_53 = (1.0 / 2.0) * d_52;
  const double d_54 = 1.0 / (d_17 * d_10);
  const double d_55 = 1.0 / d_8;
  const double d_56 = 2.0 * d_19;
  const double d_57 = 4.0 * d_13;
  const double d_58 = d_41 * d_54;
  const double d_59 = d_14 * d_58;
  const double d_60 = -d_8;
  const double d_61 = sqrt(d_10 * d_60);
  const double d_62 = 1.0 / d_61;
  const double d_63 = d_24 * r0;
  const double d_64 = sqrt(d_6 * d_60);
  const double d_65 = d_27 * d_34;
  const double d_66 = 1.0 / d_16;
  const double d_67 = 12.0 * M;
  const double d_68 = d_66 * d_67;
  const double d_69 = d_54 / d_10;
  const double d_70 = d_1 * d_1 * d_1;
  const double d_71 = d_2 * d_2 * d_2;
  const double d_72 = d_1 * d_1 * d_1 * d_1;
  const double d_73 = d_2 * d_2 * d_2 * d_2;
  const double d_74 = d_4 * d_5;
  const double d_75 = 2.0 * d_13;
  const double d_76 = d_23 * d_40 * d_46;
  const double d_77 = d_41 * d_45;
  const double d_78 = r0 * r0 * r0 * r0 * r0;
  const double d_79 = 1.0 / d_78;
  const double d_80 = 4.0 * d_79;
  const double d_81 = M * d_16;
  const double d_82 = d_26 * d_26;
  const double d_83 = (1.0 / 24.0) * d_82;
  const double d_84 = d_19 * r0;
  const double d_85 = d_12 * d_75;
  const double d_86 = d_12 * d_57;
  const double d_87 = d_13 * d_3;
  const double d_88 = d_12 * d_87;
  const double d_89 = d_4 * d_52;
  const double d_90 = d_5 * d_52;
  const double d_91 = d_4 * d_84;
  const double d_92 = d_5 * d_84;
  const double d_93 = d_36 * r0;
  const double d_94 = 1.0 / (d_25 - d_63 + d_93);
  const double d_95 = d_6 * d_94;
  const double d_96 = d_4 + d_5;
  const double d_97 = d_13 * d_96;
  const double d_98 = d_10 * d_19;
  const double d_99 = M * d_23;
  const double d_100 = d_61 * d_64;
  const double d_101 = d_64 * r0;
  const double d_102 = d_61 * d_75;
  const double d_103 = 1.0 / d_60;
  const double d_104 = d_100 * d_57;
  const double d_105 = d_100 * d_87;
  const double d_106 = d_23 * d_64;
  const double d_107 = 1.0 / (d_60 * d_60);
  const double d_108 = 1.0 / d_39;
  const double d_109 = d_16 * d_69;
  const double d_110 = d_41 * d_62;
  const double d_111 = 8.0 * d_58;
  const double d_112 = d_52 * d_64;
  const double d_113 = M * d_18;
  const double d_114 = d_110 * d_97;
  const double d_115 = d_41 * d_69;
  const double d_116 = d_115 * d_27;
  const double d_117 = 12.0 * d_27;
  const double d_118 = d_117 * d_34;
  const double d_119 = M * r0;
  const double d_120 = d_103 * d_96;
  const double d_121 = d_10 * d_85;
  const double d_122 = d_18 * d_19;
  const double d_123 = -d_1 * d_44;
  const double d_124 = d_2 * d_47;
  const double d_125 = d_123 + d_124 * d_45;
  const double d_126 = d_123 + d_124 * d_62;
  const double d_127 = d_10 * d_64;
  const double d_128 = d_1 * d_19;
  const double d_129 = d_1 * d_47 * d_62 + d_2 * d_44;
  const double d_130 = d_19 * d_2;
  const double d_131 = d_25 * d_4;
  const double d_132 = d_25 * d_5;
  const double d_133 = d_16 * d_24;
  const double d_134 =
      d_131 * d_18 + d_132 * d_18 - d_133 * d_4 - d_133 * d_5 + d_17;
  const double d_135 = -6.0 * M * d_23 * d_96 + d_10 * d_25 * d_96 + d_16;
  const double d_136 = 8.0 * d_120 * d_48 * d_64;
  const double d_137 = -d_135;
  DataVector& dv_0 = temps.at(0);
  dv_0 = Dx * Dx;
  DataVector& dv_1 = temps.at(1);
  dv_1 = Dy * Dy;
  DataVector& dv_2 = temps.at(2);
  dv_2 = -dv_1;
  DataVector& dv_3 = temps.at(3);
  dv_3 = dv_0 + dv_2;
  DataVector& dv_4 = temps.at(4);
  dv_4 = Dy * d_4;
  DataVector& dv_5 = temps.at(5);
  dv_5 = Dx * dv_4;
  DataVector& dv_6 = temps.at(6);
  dv_6 = Dx * d_5;
  DataVector& dv_7 = temps.at(7);
  dv_7 = Dy * dv_6;
  DataVector& dv_8 = temps.at(8);
  dv_8 = dv_5 - dv_7;
  DataVector& dv_9 = temps.at(9);
  dv_9 = d_3 * dv_3 + dv_8;
  DataVector& dv_10 = temps.at(10);
  dv_10 = Dx * d_1;
  DataVector& dv_11 = temps.at(11);
  dv_11 = Dy * d_2;
  DataVector& dv_12 = temps.at(12);
  dv_12 = dv_10 * dv_11;
  DataVector& dv_13 = temps.at(13);
  dv_13 = 2.0 * dv_12;
  DataVector& dv_14 = temps.at(14);
  dv_14 = -dv_13;
  DataVector& dv_15 = temps.at(15);
  dv_15 = 4.0 * dv_0;
  DataVector& dv_16 = temps.at(16);
  dv_16 = 5.0 * dv_1;
  DataVector& dv_17 = temps.at(17);
  dv_17 = Dz * Dz;
  DataVector& dv_18 = temps.at(18);
  dv_18 = 6.0 * dv_17;
  DataVector& dv_19 = temps.at(19);
  dv_19 = 5.0 * dv_0;
  DataVector& dv_20 = temps.at(20);
  dv_20 = 4.0 * dv_1;
  DataVector& dv_21 = temps.at(21);
  dv_21 = d_4 * (dv_18 + dv_19 + dv_20) + d_5 * (dv_15 + dv_16 + dv_18) + dv_14;
  DataVector& dv_22 = temps.at(22);
  dv_22 = M * dv_21;
  DataVector& dv_23 = temps.at(23);
  dv_23 = dv_1 + dv_17;
  DataVector& dv_24 = temps.at(24);
  dv_24 = dv_0 + dv_23;
  DataVector& dv_25 = temps.at(25);
  dv_25 = 10.0 * dv_12;
  DataVector& dv_26 = temps.at(26);
  dv_26 = 6.0 * dv_1;
  DataVector& dv_27 = temps.at(27);
  dv_27 = 9.0 * dv_17;
  DataVector& dv_28 = temps.at(28);
  dv_28 = 6.0 * dv_0;
  DataVector& dv_29 = temps.at(29);
  dv_29 = d_4 * (dv_1 + dv_27 + dv_28) + d_5 * (dv_0 + dv_26 + dv_27) - dv_25;
  DataVector& dv_30 = temps.at(30);
  dv_30 = d_19 * dv_29;
  DataVector& dv_31 = temps.at(31);
  dv_31 = dv_10 + dv_11;
  DataVector& dv_32 = temps.at(32);
  dv_32 = dv_31 * dv_31;
  DataVector& dv_33 = temps.at(33);
  dv_33 = d_10 * dv_32;
  DataVector& dv_34 = temps.at(34);
  dv_34 = d_17 * dv_24 + d_18 * dv_30 + d_21 * dv_33;
  DataVector& dv_35 = temps.at(35);
  dv_35 = -d_16 * dv_22 + dv_34;
  DataVector& dv_36 = temps.at(36);
  dv_36 = -d_12 * d_15 * dv_9 + dv_35;
  DataVector& dv_37 = temps.at(9);
  dv_37 = -d_11 * dv_9;
  DataVector& dv_38 = temps.at(37);
  dv_38 = d_9 * dv_37;
  DataVector& dv_39 = temps.at(24);
  dv_39 = d_16 * dv_24;
  DataVector& dv_40 = temps.at(22);
  dv_40 = -d_23 * dv_22;
  DataVector& dv_41 = temps.at(38);
  dv_41 = dv_32 * r0;
  DataVector& dv_42 = temps.at(39);
  dv_42 = d_21 * dv_41;
  DataVector& dv_43 = temps.at(30);
  dv_43 = d_10 * dv_30 + dv_39 + dv_40 + dv_42;
  DataVector& dv_44 = temps.at(40);
  dv_44 = -6.0 * Dx * Dy * d_1 * d_2;
  DataVector& dv_45 = temps.at(41);
  dv_45 = -dv_0;
  DataVector& dv_46 = temps.at(42);
  dv_46 = 2.0 * dv_1;
  DataVector& dv_47 = temps.at(43);
  dv_47 = 2.0 * dv_17;
  DataVector& dv_48 = temps.at(44);
  dv_48 = dv_45 + dv_46 + dv_47;
  DataVector& dv_49 = temps.at(45);
  dv_49 = 2.0 * dv_0;
  DataVector& dv_50 = temps.at(43);
  dv_50 = dv_2 + dv_47 + dv_49;
  DataVector& dv_51 = temps.at(46);
  dv_51 = d_35 * dv_31;
  DataVector& dv_52 = temps.at(47);
  dv_52 = -d_44 * dv_31;
  DataVector& dv_53 = temps.at(48);
  dv_53 = Dx * d_2;
  DataVector& dv_54 = temps.at(49);
  dv_54 = Dy * d_1;
  DataVector& dv_55 = temps.at(50);
  dv_55 = -dv_54;
  DataVector& dv_56 = temps.at(51);
  dv_56 = dv_53 + dv_55;
  DataVector& dv_57 = temps.at(52);
  dv_57 = d_47 * dv_56;
  DataVector& dv_58 = temps.at(53);
  dv_58 = d_45 * dv_57 + dv_52;
  DataVector& dv_59 = temps.at(54);
  dv_59 = d_48 * dv_58;
  DataVector& dv_60 = temps.at(55);
  dv_60 = 4.0 * dv_12;
  DataVector& dv_61 = temps.at(56);
  dv_61 = -dv_60;
  DataVector& dv_62 = temps.at(57);
  dv_62 = dv_23 + dv_45;
  DataVector& dv_63 = temps.at(58);
  dv_63 = dv_17 + dv_3;
  DataVector& dv_64 = temps.at(59);
  dv_64 = d_4 * dv_63;
  DataVector& dv_65 = temps.at(60);
  dv_65 = -d_5 * dv_62 - dv_61 - dv_64;
  DataVector& dv_66 = temps.at(61);
  dv_66 = 2.0 * dv_65;
  DataVector& dv_67 = temps.at(62);
  dv_67 = dv_51 * (-d_4 * dv_50 - d_5 * dv_48 - dv_44) - dv_59 * dv_66;
  DataVector& dv_68 = temps.at(63);
  dv_68 = dv_58 * dv_58;
  DataVector& dv_69 = temps.at(64);
  dv_69 = d_51 * dv_46;
  DataVector& dv_70 = temps.at(65);
  dv_70 = d_4 * dv_69 + d_50 * dv_0 + d_51 * dv_60 + dv_23;
  DataVector& dv_71 = temps.at(66);
  dv_71 = dv_68 + dv_70;
  DataVector& dv_72 = temps.at(67);
  dv_72 = 1.0 / dv_71;
  DataVector& dv_73 = temps.at(68);
  dv_73 = d_26 * dv_72;
  DataVector& dv_74 = temps.at(38);
  dv_74 = d_56 * dv_41;
  DataVector& dv_75 = temps.at(14);
  dv_75 = d_52 * (d_4 * (dv_1 + dv_49) + d_5 * (dv_0 + dv_46) + dv_14);
  DataVector& dv_76 = temps.at(69);
  dv_76 = dv_23 - dv_49;
  DataVector& dv_77 = temps.at(42);
  dv_77 = d_4 * (dv_0 + dv_17 - dv_46);
  DataVector& dv_78 = temps.at(9);
  dv_78 = d_23 * d_9 * (-d_5 * dv_76 - dv_44 - dv_77) - d_57 * dv_37 +
          d_9 * dv_74 + d_9 * dv_75;
  DataVector& dv_79 = temps.at(35);
  dv_79 = d_15 * dv_38 + dv_35;
  DataVector& dv_80 = temps.at(40);
  dv_80 = d_28 * dv_79;
  DataVector& dv_81 = temps.at(70);
  dv_81 = sqrt(dv_80);
  DataVector& dv_82 = temps.at(71);
  dv_82 = 6.0 * dv_12;
  DataVector& dv_83 = temps.at(43);
  dv_83 = -d_4 * dv_50 - d_5 * dv_48 + dv_82;
  DataVector& dv_84 = temps.at(52);
  dv_84 = d_62 * dv_57 + dv_52;
  DataVector& dv_85 = temps.at(47);
  dv_85 = d_48 * dv_84;
  DataVector& dv_86 = temps.at(44);
  dv_86 = -dv_62;
  DataVector& dv_87 = temps.at(59);
  dv_87 = d_5 * dv_86 + dv_60 - dv_64;
  DataVector& dv_88 = temps.at(72);
  dv_88 = 2.0 * dv_87;
  DataVector& dv_89 = temps.at(73);
  dv_89 = dv_51 * dv_83 - dv_85 * dv_88;
  DataVector& dv_90 = temps.at(74);
  dv_90 = dv_89 * dv_89;
  DataVector& dv_91 = temps.at(75);
  dv_91 = 1.0 / (dv_71 * dv_71);
  DataVector& dv_92 = temps.at(76);
  dv_92 = -dv_3;
  DataVector& dv_93 = temps.at(77);
  dv_93 = d_61 * (d_3 * dv_92 - dv_5 + dv_7);
  DataVector& dv_94 = temps.at(78);
  dv_94 = d_64 * dv_93;
  DataVector& dv_95 = temps.at(79);
  dv_95 = d_30 * dv_94;
  DataVector& dv_96 = temps.at(80);
  dv_96 = d_31 * (dv_43 + dv_95);
  DataVector& dv_97 = temps.at(81);
  dv_97 = d_56 * (dv_31 * dv_31 * dv_31 * dv_31);
  DataVector& dv_98 = temps.at(82);
  dv_98 = 3.0 * dv_0;
  DataVector& dv_99 = temps.at(83);
  dv_99 = 4.0 * dv_17;
  DataVector& dv_100 = temps.at(84);
  dv_100 = dv_20 + dv_99;
  DataVector& dv_101 = temps.at(85);
  dv_101 = dv_100 - dv_98;
  DataVector& dv_102 = temps.at(86);
  dv_102 = 3.0 * dv_1;
  DataVector& dv_103 = temps.at(87);
  dv_103 = dv_15 + dv_99;
  DataVector& dv_104 = temps.at(88);
  dv_104 = -dv_102 + dv_103;
  DataVector& dv_105 = temps.at(89);
  dv_105 = M * dv_32;
  DataVector& dv_106 = temps.at(90);
  dv_106 = dv_105 * r0;
  DataVector& dv_107 = temps.at(91);
  dv_107 = Dx * d_70;
  DataVector& dv_108 = temps.at(92);
  dv_108 = 30.0 * dv_107 * dv_11;
  DataVector& dv_109 = temps.at(93);
  dv_109 = Dy * d_71;
  DataVector& dv_110 = temps.at(94);
  dv_110 = 30.0 * dv_10 * dv_109;
  DataVector& dv_111 = temps.at(95);
  dv_111 = dv_110 * dv_63;
  DataVector& dv_112 = temps.at(96);
  dv_112 = Dx * Dx * Dx * Dx;
  DataVector& dv_113 = temps.at(97);
  dv_113 = 2.0 * dv_112;
  DataVector& dv_114 = temps.at(98);
  dv_114 = 11.0 * dv_0;
  DataVector& dv_115 = temps.at(99);
  dv_115 = dv_113 - dv_114 * dv_23 + 2.0 * (dv_23 * dv_23);
  DataVector& dv_116 = temps.at(100);
  dv_116 = 11.0 * dv_1;
  DataVector& dv_117 = temps.at(83);
  dv_117 = dv_116 - dv_99;
  DataVector& dv_118 = temps.at(101);
  dv_118 = Dy * Dy * Dy * Dy;
  DataVector& dv_119 = temps.at(102);
  dv_119 = Dz * Dz * Dz * Dz;
  DataVector& dv_120 = temps.at(97);
  dv_120 = dv_113 - dv_116 * dv_17 + 2.0 * dv_118 + 2.0 * dv_119;
  DataVector& dv_121 = temps.at(103);
  dv_121 = 68.0 * dv_1;
  DataVector& dv_122 = temps.at(104);
  dv_122 = 7.0 * dv_17;
  DataVector& dv_123 = temps.at(105);
  dv_123 = dv_121 - dv_122;
  DataVector& dv_124 = temps.at(101);
  dv_124 = dv_1 * dv_122 + 11.0 * dv_112 + 11.0 * dv_118 - 4.0 * dv_119;
  DataVector& dv_125 = temps.at(96);
  dv_125 = dv_31 * dv_31 * dv_31;
  DataVector& dv_126 = temps.at(102);
  dv_126 = -dv_56;
  DataVector& dv_127 = temps.at(33);
  dv_127 = d_75 * dv_33;
  DataVector& dv_128 = temps.at(106);
  dv_128 = d_61 * dv_31;
  DataVector& dv_129 = temps.at(107);
  dv_129 = d_42 * dv_87 * r0;
  DataVector& dv_130 = temps.at(108);
  dv_130 = 3.0 * dv_17;
  DataVector& dv_131 = temps.at(86);
  dv_131 = d_14 * (d_4 * (-dv_130 - dv_2 - dv_98) +
                   d_5 * (-dv_102 - dv_130 - dv_45) + 8.0 * dv_12);
  DataVector& dv_132 = temps.at(2);
  dv_132 = d_76 * dv_83;
  DataVector& dv_133 = temps.at(41);
  dv_133 = -d_40 * dv_126 * dv_127 + d_56 * d_61 * dv_125 - dv_126 * dv_132 +
           dv_128 * dv_129 + dv_128 * dv_131;
  DataVector& dv_134 = temps.at(108);
  dv_134 = d_21 * dv_31;
  DataVector& dv_135 = temps.at(8);
  dv_135 = d_3 * dv_0 - d_3 * dv_1 + dv_8;
  DataVector& dv_136 = temps.at(55);
  dv_136 = -d_52 * dv_135 + 5.0 * d_84 * dv_135 +
           d_85 * (d_4 * dv_3 - d_5 * dv_3 - dv_60) - dv_134 * dv_56;
  DataVector& dv_137 = temps.at(82);
  dv_137 = d_20 * d_5 * dv_28;
  DataVector& dv_138 = temps.at(109);
  dv_138 = d_20 * d_4 * dv_26;
  DataVector& dv_139 = temps.at(110);
  dv_139 = d_5 * dv_0;
  DataVector& dv_140 = temps.at(111);
  dv_140 = d_4 * dv_1;
  DataVector& dv_141 = temps.at(18);
  dv_141 = d_52 * dv_18;
  DataVector& dv_142 = temps.at(112);
  dv_142 = 12.0 * d_20 * dv_12;
  DataVector& dv_143 = temps.at(16);
  dv_143 = d_23 * dv_0 + d_23 * dv_1 + d_23 * dv_17 - d_4 * dv_141 -
           d_5 * dv_141 + d_52 * dv_13 + d_84 * dv_139 + d_84 * dv_140 -
           d_84 * dv_25 - d_89 * dv_19 - d_89 * dv_20 - d_90 * dv_15 -
           d_90 * dv_16 + d_91 * dv_27 + d_91 * dv_28 + d_92 * dv_26 +
           d_92 * dv_27 + dv_137 + dv_138 + dv_142;
  DataVector& dv_manual = get(get<CurvedScalarWave::Tags::Psi>(*result));
  dv_manual = d_95 * (-d_86 * dv_5 + d_86 * dv_7 - d_88 * dv_15 + d_88 * dv_20 +
                      dv_143);
  DataVector& dv_144 = temps.at(19);
  dv_144 = 1. / (dv_manual * sqrt(dv_manual));
  DataVector& dv_145 = temps.at(25);
  dv_145 = d_27 * dv_144;
  DataVector& dv_146 = temps.at(18);
  dv_146 = 4.0 * dv_53;
  DataVector& dv_147 = temps.at(113);
  dv_147 = 5.0 * dv_54;
  DataVector& dv_148 = temps.at(114);
  dv_148 = 5.0 * dv_53;
  DataVector& dv_149 = temps.at(115);
  dv_149 = 4.0 * dv_54;
  DataVector& dv_150 = temps.at(116);
  dv_150 = d_5 * dv_53;
  DataVector& dv_151 = temps.at(117);
  dv_151 = -d_4 * dv_54 + dv_150;
  DataVector& dv_152 = temps.at(118);
  dv_152 = d_4 * (dv_148 - dv_149) + d_5 * (dv_146 - dv_147) + dv_151;
  DataVector& dv_153 = temps.at(113);
  dv_153 = -d_4 * dv_147 + d_4 * (6 * dv_53 + dv_55) + d_5 * dv_148 +
           d_5 * (dv_53 - 6.0 * dv_54);
  DataVector& dv_154 = temps.at(114);
  dv_154 = dv_53 + dv_54;
  DataVector& dv_155 = temps.at(91);
  dv_155 = 2.0 * d_3 * dv_154 - d_4 * dv_10 - d_5 * dv_11 + dv_107 + dv_109;
  DataVector& dv_156 = temps.at(93);
  dv_156 = d_85 * dv_155;
  DataVector& dv_157 = temps.at(119);
  dv_157 = -d_10 * dv_156 + d_17 * dv_56 + d_18 * d_19 * dv_153 - d_81 * dv_152;
  DataVector& dv_158 = temps.at(120);
  dv_158 = Dx * d_71;
  DataVector& dv_159 = temps.at(121);
  dv_159 = -Dy * d_70 + dv_151 + dv_158;
  DataVector& dv_160 = temps.at(122);
  dv_160 = -2.0 * Dy * d_1 * d_4 + d_4 * dv_154 - d_5 * dv_154 + 2.0 * dv_150;
  DataVector& dv_161 = temps.at(27);
  dv_161 = d_98 * dv_27;
  DataVector& dv_162 = temps.at(110);
  dv_162 = -10.0 * Dx * Dy * d_1 * d_10 * d_19 * d_2 -
           4.0 * Dx * Dy * d_13 * d_4 * d_61 * d_64 * r0 -
           5.0 * M * d_23 * d_4 * dv_0 - 4.0 * M * d_23 * d_4 * dv_1 -
           6.0 * M * d_23 * d_4 * dv_17 - 4.0 * M * d_23 * d_5 * dv_0 -
           5.0 * M * d_23 * d_5 * dv_1 - 6.0 * M * d_23 * d_5 * dv_17 -
           4.0 * d_1 * d_13 * d_2 * d_61 * d_64 * dv_0 * r0 +
           d_100 * d_30 * dv_7 + d_101 * d_61 * d_87 * dv_20 + d_16 * dv_0 +
           d_16 * dv_1 + d_16 * dv_17 + d_4 * d_98 * dv_28 + d_4 * dv_161 +
           d_5 * d_98 * dv_26 + d_5 * dv_161 + d_98 * dv_139 + d_98 * dv_140 +
           d_99 * dv_13 + dv_137 * r0 + dv_138 * r0 + dv_142 * r0;
  DataVector& dv_163 = temps.at(13);
  dv_163 = sqrt(d_33 * dv_162);
  DataVector& dv_164 = temps.at(109);
  dv_164 = 1.0 / dv_79;
  DataVector& dv_165 = temps.at(28);
  dv_165 = dv_163 * dv_164;
  DataVector& dv_166 = temps.at(111);
  dv_166 = dv_165 * dv_73;
  DataVector& dv_167 = temps.at(27);
  dv_167 = -Dx * d_2 * d_50 - d_46 * d_62 * d_96 * dv_84 * r0 + d_49 * dv_150 +
           dv_54;
  DataVector& dv_168 = temps.at(82);
  dv_168 = 6.0 * dv_32;
  DataVector& dv_169 = temps.at(26);
  dv_169 = d_35 * dv_83;
  DataVector& dv_170 = temps.at(8);
  dv_170 = -dv_135;
  DataVector& dv_171 = temps.at(112);
  dv_171 = d_47 * dv_31;
  DataVector& dv_172 = temps.at(123);
  dv_172 = d_44 * dv_56;
  DataVector& dv_173 = temps.at(72);
  dv_173 = d_48 * dv_88;
  DataVector& dv_174 = temps.at(124);
  dv_174 = d_52 * dv_89;
  DataVector& dv_175 = temps.at(28);
  dv_175 = d_26 * dv_165 * dv_174 * dv_91;
  DataVector& dv_176 = temps.at(125);
  dv_176 = 1.0 / (dv_79 * dv_79);
  DataVector& dv_177 = temps.at(126);
  dv_177 = dv_163 * dv_176;
  DataVector& dv_178 = temps.at(127);
  dv_178 = dv_177 * dv_73 * dv_89;
  DataVector& dv_179 = temps.at(128);
  dv_179 = dv_134 * r0;
  DataVector& dv_180 = temps.at(129);
  dv_180 = 1.0 / dv_163;
  DataVector& dv_181 = temps.at(130);
  dv_181 = d_99 * dv_152;
  DataVector& dv_182 = temps.at(131);
  dv_182 = d_19 * dv_153;
  DataVector& dv_183 = temps.at(124);
  dv_183 = dv_174 * dv_73;
  DataVector& dv_184 = temps.at(132);
  dv_184 = 1.0 / (dv_79 * dv_79 * dv_79);
  DataVector& dv_185 = temps.at(16);
  dv_185 = d_95 * (-d_104 * dv_5 + d_104 * dv_7 - d_105 * dv_15 +
                   d_105 * dv_20 + dv_143);
  DataVector& dv_186 = temps.at(133);
  dv_186 = dv_185 * sqrt(dv_185);
  DataVector& dv_187 = temps.at(134);
  dv_187 = d_103 * dv_186;
  DataVector& dv_188 = temps.at(71);
  dv_188 = d_106 * (-d_5 * dv_76 - dv_77 + dv_82) - d_57 * dv_93 +
           d_64 * dv_74 + d_64 * dv_75;
  DataVector& dv_189 = temps.at(77);
  dv_189 = d_59 * dv_188;
  DataVector& dv_190 = temps.at(16);
  dv_190 = sqrt(dv_185);
  DataVector& dv_191 = temps.at(69);
  dv_191 = dv_84 * dv_84;
  DataVector& dv_192 = temps.at(38);
  dv_192 = dv_190 * dv_191;
  DataVector& dv_193 = temps.at(42);
  dv_193 = d_103 * dv_192;
  DataVector& dv_194 = temps.at(14);
  dv_194 = 1.0 / dv_190;
  DataVector& dv_195 = temps.at(135);
  dv_195 = -dv_89;
  DataVector& dv_196 = temps.at(136);
  dv_196 = dv_195 * dv_195;
  DataVector& dv_197 = temps.at(137);
  dv_197 = dv_194 * dv_196;
  DataVector& dv_198 = temps.at(79);
  dv_198 = -d_10 * d_19 * dv_29 - dv_39 - dv_40 - dv_42 - dv_95;
  DataVector& dv_199 = temps.at(65);
  dv_199 = dv_191 + dv_70;
  DataVector& dv_200 = temps.at(29);
  dv_200 = 1.0 / (dv_199 * dv_199);
  DataVector& dv_201 = temps.at(34);
  dv_201 = d_15 * dv_94 - d_81 * dv_21 + dv_34;
  DataVector& dv_202 = temps.at(78);
  dv_202 = dv_200 * dv_201;
  DataVector& dv_203 = temps.at(21);
  dv_203 = d_107 * dv_202;
  DataVector& dv_204 = temps.at(22);
  dv_204 = d_65 * dv_203;
  DataVector& dv_205 = temps.at(39);
  dv_205 = dv_198 * dv_204;
  DataVector& dv_206 = temps.at(110);
  dv_206 = sqrt(d_107 * d_32 * dv_162);
  DataVector& dv_207 = temps.at(12);
  dv_207 = -d_4 * dv_104 - d_5 * dv_101 + 14.0 * dv_12;
  DataVector& dv_208 = temps.at(24);
  dv_208 = -dv_117;
  DataVector& dv_209 = temps.at(138);
  dv_209 = -dv_123;
  DataVector& dv_210 = temps.at(139);
  dv_210 = -dv_133;
  DataVector& dv_211 = temps.at(140);
  dv_211 = d_110 * dv_84;
  DataVector& dv_212 = temps.at(141);
  dv_212 = d_108 * d_68 * (dv_87 * dv_87) +
           d_109 * (d_14 * (d_72 * dv_115 + d_73 * (dv_0 * dv_208 + dv_120) +
                            d_74 * (-dv_0 * dv_209 - dv_124) + dv_108 * dv_86 -
                            dv_111) +
                    dv_106 * dv_207 + dv_97) +
           d_80 * dv_210 * dv_211;
  DataVector& dv_213 = temps.at(142);
  dv_213 = -d_7 * dv_90 + dv_199 * dv_212;
  DataVector& dv_214 = temps.at(143);
  dv_214 = dv_206 * dv_213;
  DataVector& dv_215 = temps.at(144);
  dv_215 = d_28 * dv_214;
  DataVector& dv_216 = temps.at(77);
  dv_216 = d_63 * dv_197 * dv_205 - dv_187 * dv_189 + dv_189 * dv_193 +
           dv_202 * dv_215;
  DataVector& dv_217 = temps.at(134);
  dv_217 = d_111 * dv_187;
  DataVector& dv_218 = temps.at(91);
  dv_218 = -d_102 * dv_155;
  DataVector& dv_219 = temps.at(145);
  dv_219 = 2.0 * dv_54;
  DataVector& dv_220 = temps.at(146);
  dv_220 = 2.0 * dv_53;
  DataVector& dv_221 = temps.at(147);
  dv_221 = dv_219 + dv_53;
  DataVector& dv_222 = temps.at(148);
  dv_222 = dv_220 + dv_54;
  DataVector& dv_223 = temps.at(149);
  dv_223 = -3.0 * Dy * d_1 * d_4 + 3.0 * dv_150;
  DataVector& dv_224 = temps.at(146);
  dv_224 =
      d_113 *
      (d_106 * (-d_4 * dv_221 + d_5 * dv_222 - dv_223) +
       d_112 * (d_4 * (dv_220 + dv_55) + d_5 * (-dv_219 + dv_53) + dv_151) -
       dv_218);
  DataVector& dv_225 = temps.at(117);
  dv_225 = d_103 * dv_188 * dv_190;
  DataVector& dv_226 = temps.at(42);
  dv_226 = d_111 * dv_193;
  DataVector& dv_227 = temps.at(91);
  dv_227 = d_64 * dv_218;
  DataVector& dv_228 = temps.at(118);
  dv_228 = d_10 * dv_227 + d_17 * dv_56 + d_18 * dv_182 - d_81 * dv_152;
  DataVector& dv_229 = temps.at(50);
  dv_229 = d_116 * dv_228;
  DataVector& dv_230 = temps.at(145);
  dv_230 = d_103 * dv_188 * dv_191 * dv_194;
  DataVector& dv_231 = temps.at(150);
  dv_231 = dv_200 * dv_228;
  DataVector& dv_232 = temps.at(151);
  dv_232 = d_98 * dv_197;
  DataVector& dv_233 = temps.at(152);
  dv_233 = d_107 * dv_198;
  DataVector& dv_234 = temps.at(153);
  dv_234 = d_118 * dv_233;
  DataVector& dv_235 = temps.at(91);
  dv_235 = d_16 * dv_126 - d_98 * dv_153 + dv_181 - dv_227 * r0;
  DataVector& dv_236 = temps.at(113);
  dv_236 = d_118 * dv_203;
  DataVector& dv_237 = temps.at(154);
  dv_237 = -dv_167;
  DataVector& dv_238 = temps.at(155);
  dv_238 = dv_201 * 1.0 / (dv_199 * dv_199 * dv_199);
  DataVector& dv_239 = temps.at(152);
  dv_239 = 24.0 * d_65 * dv_233 * dv_238;
  DataVector& dv_240 = temps.at(21);
  dv_240 = d_54 * dv_196 * dv_198 * dv_203 * 1.0 / d_82 * 1.0 / dv_186;
  DataVector& dv_241 = temps.at(156);
  dv_241 = -dv_160;
  DataVector& dv_242 = temps.at(157);
  dv_242 = 2.0 * M * d_16 * d_34 * dv_159 * dv_31 + d_114 * dv_87 +
           d_44 * dv_241 * dv_84;
  DataVector& dv_243 = temps.at(158);
  dv_243 = d_119 * dv_194 * dv_195;
  DataVector& dv_244 = temps.at(39);
  dv_244 = dv_205 * dv_243;
  DataVector& dv_245 = temps.at(143);
  dv_245 = d_27 * dv_214;
  DataVector& dv_246 = temps.at(155);
  dv_246 = 4.0 * dv_238;
  DataVector& dv_247 = temps.at(22);
  dv_247 = dv_204 * dv_213 * 1.0 / dv_206;
  DataVector& dv_248 = temps.at(156);
  dv_248 = M * dv_241;
  DataVector& dv_249 = temps.at(159);
  dv_249 = d_108 * dv_87;
  DataVector& dv_250 = temps.at(160);
  dv_250 = 24.0 * dv_249;
  DataVector& dv_251 = temps.at(139);
  dv_251 = 2.0 * d_79 * dv_210;
  DataVector& dv_252 = temps.at(161);
  dv_252 = 3.0 * dv_54;
  DataVector& dv_253 = temps.at(162);
  dv_253 = 3.0 * dv_53;
  DataVector& dv_254 = temps.at(58);
  dv_254 = 15.0 * dv_63;
  DataVector& dv_255 = temps.at(163);
  dv_255 = dv_254 * dv_54;
  DataVector& dv_256 = temps.at(44);
  dv_256 = 15.0 * dv_53 * dv_86;
  DataVector& dv_257 = temps.at(164);
  dv_257 = 15.0 * dv_62;
  DataVector& dv_258 = temps.at(165);
  dv_258 = d_2 * (Dx * Dx * Dx);
  DataVector& dv_259 = temps.at(166);
  dv_259 = dv_114 * dv_54 + 4.0 * dv_258;
  DataVector& dv_260 = temps.at(167);
  dv_260 = Dy * Dy * Dy;
  DataVector& dv_261 = temps.at(168);
  dv_261 = 11.0 * dv_17;
  DataVector& dv_262 = temps.at(33);
  dv_262 = d_40 * dv_127;
  DataVector& dv_263 = temps.at(169);
  dv_263 = d_29 * dv_128;
  DataVector& dv_264 = temps.at(170);
  dv_264 = d_10 * dv_128;
  DataVector& dv_265 = temps.at(171);
  dv_265 = 8.0 * dv_264;
  DataVector& dv_266 = temps.at(140);
  dv_266 = 2.0 * dv_211;
  DataVector& dv_267 = temps.at(78);
  dv_267 = d_28 * dv_202 * dv_206;
  DataVector& dv_268 = temps.at(125);
  dv_268 = d_83 * dv_176;
  DataVector& dv_269 = temps.at(172);
  dv_269 = Dx * d_4;
  DataVector& dv_270 = temps.at(173);
  dv_270 = d_1 * dv_11;
  DataVector& dv_271 = temps.at(174);
  dv_271 = -dv_270;
  DataVector& dv_272 = temps.at(175);
  dv_272 = 5.0 * dv_269 + dv_271 + 4.0 * dv_6;
  DataVector& dv_273 = temps.at(176);
  dv_273 = Dy * d_5;
  DataVector& dv_274 = temps.at(177);
  dv_274 = -dv_273;
  DataVector& dv_275 = temps.at(178);
  dv_275 = d_2 * dv_10;
  DataVector& dv_276 = temps.at(179);
  dv_276 = dv_274 + 2.0 * dv_275 + dv_4;
  DataVector& dv_277 = temps.at(180);
  dv_277 = 6.0 * dv_269 - 5.0 * dv_270 + dv_6;
  DataVector& dv_278 = temps.at(108);
  dv_278 = d_10 * dv_134;
  DataVector& dv_279 = temps.at(181);
  dv_279 = Dx * d_17 + d_1 * dv_278 + d_122 * dv_277;
  DataVector& dv_280 = temps.at(182);
  dv_280 = -d_121 * dv_276 - d_81 * dv_272 + dv_279;
  DataVector& dv_281 = temps.at(183);
  dv_281 = d_28 * dv_144;
  DataVector& dv_282 = temps.at(184);
  dv_282 = -dv_269;
  DataVector& dv_283 = temps.at(185);
  dv_283 = 2.0 * dv_270 + dv_282 + dv_6;
  DataVector& dv_284 = temps.at(186);
  dv_284 = 2.0 * dv_269;
  DataVector& dv_285 = temps.at(187);
  dv_285 = 3.0 * dv_270;
  DataVector& dv_286 = temps.at(188);
  dv_286 = -dv_284 + dv_285 + dv_6;
  DataVector& dv_287 = temps.at(189);
  dv_287 = 2.0 * dv_51;
  DataVector& dv_288 = temps.at(190);
  dv_288 = d_1 * dv_169 + dv_286 * dv_287;
  DataVector& dv_289 = temps.at(191);
  dv_289 = d_53 * dv_166;
  DataVector& dv_290 = temps.at(192);
  dv_290 = Dx * d_50 + d_49 * dv_270;
  DataVector& dv_291 = temps.at(126);
  dv_291 = dv_177 * dv_183;
  DataVector& dv_292 = temps.at(193);
  dv_292 = -d_102 * dv_276;
  DataVector& dv_293 = temps.at(180);
  dv_293 = Dx * d_16 - M * d_23 * dv_272 + d_1 * dv_179 + d_10 * d_19 * dv_277 +
           d_101 * dv_292;
  DataVector& dv_294 = temps.at(194);
  dv_294 = d_1 * dv_31;
  DataVector& dv_295 = temps.at(174);
  dv_295 = d_101 * d_56 * dv_294 + d_106 * (dv_282 + dv_285 + 2.0 * dv_6) +
           d_112 * (dv_271 + dv_284 + dv_6) - dv_292;
  DataVector& dv_296 = temps.at(187);
  dv_296 = d_10 * dv_217;
  DataVector& dv_297 = temps.at(184);
  dv_297 = d_126 * dv_84;
  DataVector& dv_298 = temps.at(186);
  dv_298 = d_10 * dv_225;
  DataVector& dv_299 = temps.at(195);
  dv_299 = d_111 * dv_298;
  DataVector& dv_300 = temps.at(175);
  dv_300 = d_127 * dv_292 - d_81 * dv_272 + dv_279;
  DataVector& dv_301 = temps.at(181);
  dv_301 = d_115 * dv_300;
  DataVector& dv_302 = temps.at(186);
  dv_302 = d_117 * dv_298;
  DataVector& dv_303 = temps.at(193);
  dv_303 = d_14 * dv_230;
  DataVector& dv_304 = temps.at(196);
  dv_304 = dv_200 * dv_300;
  DataVector& dv_305 = temps.at(137);
  dv_305 = d_119 * dv_197;
  DataVector& dv_306 = temps.at(197);
  dv_306 = dv_234 * dv_305;
  DataVector& dv_307 = temps.at(198);
  dv_307 = -dv_293;
  DataVector& dv_308 = temps.at(199);
  dv_308 = dv_236 * dv_305;
  DataVector& dv_309 = temps.at(200);
  dv_309 = dv_290 + dv_297;
  DataVector& dv_310 = temps.at(137);
  dv_310 = dv_239 * dv_305;
  DataVector& dv_311 = temps.at(201);
  dv_311 = d_63 * dv_240;
  DataVector& dv_312 = temps.at(202);
  dv_312 = -d_126 * dv_173 - 4.0 * dv_283 * dv_85 + dv_288;
  DataVector& dv_313 = temps.at(203);
  dv_313 = 2.0 * dv_215;
  DataVector& dv_314 = temps.at(144);
  dv_314 = dv_215 * dv_246;
  DataVector& dv_315 = temps.at(204);
  dv_315 = dv_247 * r0;
  DataVector& dv_316 = temps.at(205);
  dv_316 = d_7 * dv_89;
  DataVector& dv_317 = temps.at(206);
  dv_317 = M * dv_283;
  DataVector& dv_318 = temps.at(207);
  dv_318 = d_66 * dv_250;
  DataVector& dv_319 = temps.at(96);
  dv_319 = 4.0 * dv_125;
  DataVector& dv_320 = temps.at(12);
  dv_320 = d_119 * dv_207;
  DataVector& dv_321 = temps.at(87);
  dv_321 = dv_103 - dv_116;
  DataVector& dv_322 = temps.at(208);
  dv_322 = d_110 * dv_251;
  DataVector& dv_323 = temps.at(209);
  dv_323 = d_61 * dv_168;
  DataVector& dv_324 = temps.at(107);
  dv_324 = d_61 * dv_129;
  DataVector& dv_325 = temps.at(86);
  dv_325 = d_61 * dv_131;
  DataVector& dv_326 = temps.at(210);
  dv_326 = d_79 * dv_266;
  DataVector& dv_327 = temps.at(211);
  dv_327 = 2.0 * dv_267;
  DataVector& dv_328 = temps.at(212);
  dv_328 = d_81 * dv_268;
  DataVector& dv_329 = temps.at(213);
  dv_329 = -dv_275;
  DataVector& dv_330 = temps.at(214);
  dv_330 = 5.0 * dv_273 + dv_329 + 4.0 * dv_4;
  DataVector& dv_331 = temps.at(215);
  dv_331 = 6.0 * dv_273 - 5.0 * dv_275 + dv_4;
  DataVector& dv_332 = temps.at(108);
  dv_332 = Dy * d_17 + d_122 * dv_331 + d_2 * dv_278;
  DataVector& dv_333 = temps.at(216);
  dv_333 = d_121 * dv_283 - d_81 * dv_330 + dv_332;
  DataVector& dv_334 = temps.at(217);
  dv_334 = 2.0 * dv_273;
  DataVector& dv_335 = temps.at(218);
  dv_335 = 3.0 * dv_275;
  DataVector& dv_336 = temps.at(219);
  dv_336 = -dv_334 + dv_335 + dv_4;
  DataVector& dv_337 = temps.at(220);
  dv_337 = d_2 * dv_169;
  DataVector& dv_338 = temps.at(221);
  dv_338 = 2.0 * dv_4;
  DataVector& dv_339 = temps.at(222);
  dv_339 = Dy + d_49 * dv_275 + d_51 * dv_338;
  DataVector& dv_340 = temps.at(223);
  dv_340 = d_102 * dv_283;
  DataVector& dv_341 = temps.at(215);
  dv_341 = Dy * d_16 - M * d_23 * dv_330 + d_10 * d_19 * dv_331 +
           d_101 * dv_340 + d_2 * dv_179;
  DataVector& dv_342 = temps.at(221);
  dv_342 = M * d_10 * d_64 * (dv_329 + dv_334 + dv_4) -
           d_106 * (-dv_274 - dv_335 - dv_338) +
           2.0 * d_19 * d_2 * d_64 * dv_31 * r0 - dv_340;
  DataVector& dv_343 = temps.at(213);
  dv_343 = d_129 * dv_84;
  DataVector& dv_344 = temps.at(223);
  dv_344 = d_127 * dv_340 - d_81 * dv_330 + dv_332;
  DataVector& dv_345 = temps.at(108);
  dv_345 = d_115 * dv_302;
  DataVector& dv_346 = temps.at(214);
  dv_346 = -dv_341;
  DataVector& dv_347 = temps.at(177);
  dv_347 = dv_339 - dv_343;
  DataVector& dv_348 = temps.at(59);
  dv_348 = -2.0 * d_129 * d_22 * d_41 * dv_87 + 4.0 * dv_276 * dv_85 -
           dv_287 * dv_336 - dv_337;
  DataVector& dv_349 = temps.at(217);
  dv_349 = M * dv_276;
  DataVector& dv_350 = temps.at(218);
  dv_350 = 30.0 * dv_1;
  DataVector& dv_351 = temps.at(10);
  dv_351 = d_71 * dv_10;
  DataVector& dv_352 = temps.at(84);
  dv_352 = d_72 * (dv_100 - dv_114);
  DataVector& dv_353 = temps.at(224);
  dv_353 = -d_22 * d_41 * dv_84 + dv_51;

  get(get<CurvedScalarWave::Tags::Psi>(*result)) =
      d_53 * dv_67 * dv_73 * sqrt(d_33 * (d_30 * dv_38 + dv_43)) * 1.0 / dv_36 -
      d_81 * d_83 * 1.0 / (dv_36 * dv_36) *
          (4 * d_10 * d_41 * d_54 * d_55 * dv_78 * dv_80 * sqrt(dv_80) +
           d_22 * d_27 * dv_79 * dv_91 * sqrt(d_32 * dv_96) *
               (-d_7 * dv_67 * dv_67 +
                dv_71 * (d_16 * d_69 *
                             (d_14 * (d_72 * dv_115 +
                                      d_73 * (-dv_0 * dv_117 + dv_120) -
                                      d_74 * (-dv_0 * dv_123 + dv_124) -
                                      dv_108 * dv_62 - dv_111) +
                              dv_106 * (14 * Dx * Dy * d_1 * d_2 -
                                        d_4 * dv_104 - d_5 * dv_101) +
                              dv_97) -
                         d_68 * 1.0 / d_38 * dv_65 * dv_65 -
                         d_77 * d_80 * dv_133 * dv_58)) -
           d_55 * d_59 * dv_68 * dv_78 * dv_81 -
           d_63 * d_65 * dv_79 * dv_90 * dv_91 * dv_96 * 1.0 / dv_81) +
      1.0 / sqrt(d_28 * dv_36);
  get(get<::Tags::dt<CurvedScalarWave::Tags::Psi>>(*result)) =
      w *
      ((1.0 / 2.0) * M * d_10 * d_26 * dv_163 * dv_164 * dv_72 *
           (d_35 * dv_126 * dv_168 + dv_126 * dv_169 - 8.0 * dv_170 * dv_85 -
            dv_173 * (d_62 * dv_171 + dv_172)) +
       M * d_18 * d_26 * dv_163 * dv_164 * dv_167 * dv_89 * dv_91 -
       M * d_18 * dv_157 * dv_178 +
       (1.0 / 2.0) * M * d_26 * d_31 * d_37 * dv_164 * dv_180 * dv_72 * dv_89 *
           r0 *
           (d_101 * d_102 * (d_4 * dv_3 + d_5 * dv_92 + dv_61) -
            5.0 * d_98 * dv_170 + d_99 * dv_170 + dv_126 * dv_179) +
       (1.0 / 6.0) * M * d_78 * d_82 * dv_157 * dv_184 * dv_216 -
       d_10 * dv_166 *
           (2 * d_34 * d_81 * dv_159 * dv_31 - d_44 * dv_160 * dv_58 +
            d_77 * d_97 * dv_65) -
       d_16 * dv_268 *
           (-M * d_6 * dv_237 * dv_245 * dv_246 +
            4.0 * d_113 * dv_229 * dv_230 +
            8.0 * d_114 * d_16 * d_54 * dv_225 * dv_84 -
            d_18 * d_67 * dv_225 * dv_229 +
            d_42 * dv_267 *
                (dv_199 *
                     (d_120 * d_41 * d_46 * dv_251 +
                      d_66 * dv_266 *
                          (2 * d_23 * d_40 * d_46 * dv_126 *
                               (-d_4 * dv_222 + d_5 * dv_221 - dv_223) -
                           d_96 * dv_132 - d_96 * dv_262 - dv_248 * dv_263 -
                           dv_265 *
                               (4 * Dy * d_1 * d_4 - d_4 * (dv_253 + dv_54) -
                                d_5 * dv_146 + d_5 * (dv_252 + dv_53))) +
                      d_69 *
                          (d_29 * (-Dy * d_4 * d_70 * dv_257 +
                                   d_5 * dv_158 * dv_254 - d_72 * dv_256 +
                                   d_72 * (-dv_149 * dv_23 -
                                           11.0 * dv_23 * dv_53 + dv_259) -
                                   d_73 * dv_255 +
                                   d_73 * (-4 * d_1 * dv_260 + dv_208 * dv_53 +
                                           dv_259 + dv_261 * dv_54) +
                                   d_74 * (7 * Dy * d_1 * dv_17 +
                                           22.0 * d_1 * dv_260 -
                                           68.0 * dv_0 * dv_54 -
                                           dv_209 * dv_53 - 22.0 * dv_258) +
                                   dv_108 * dv_154 - dv_110 * dv_154) +
                           dv_105 *
                               (7 * Dy * d_1 * d_4 - d_4 * (dv_146 + dv_252) +
                                d_5 * (dv_149 + dv_253) - 7.0 * dv_150)) *
                          (r0 * r0 * r0 * r0 * r0 * r0) +
                      dv_248 * dv_250 * 1.0 / d_18) +
                 dv_212 * dv_237 * r0 + 6.0 * dv_242 * dv_89) +
            d_43 * dv_231 * dv_245 - d_52 * dv_235 * dv_247 -
            6.0 * d_98 * dv_228 * dv_240 - dv_217 * dv_224 + dv_224 * dv_226 +
            dv_231 * dv_232 * dv_234 + dv_232 * dv_235 * dv_236 -
            dv_232 * dv_237 * dv_239 + 24.0 * dv_242 * dv_244) -
       1.0 / 2.0 * d_31 * d_37 * dv_164 * dv_180 * dv_183 *
           (-d_10 * dv_182 - d_16 * dv_56 + dv_156 * r0 + dv_181) -
       d_6 * dv_145 * dv_157 - d_81 * dv_136 * dv_178 - dv_136 * dv_145 -
       dv_175 * (-d_3 * d_51 * dv_49 + d_3 * dv_69 - d_49 * dv_5 + d_49 * dv_7 +
                 dv_58 * (d_45 * dv_171 + dv_172)));
  get<0>(get<::Tags::deriv<CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                           Frame::Inertial>>(*result)) =
      (1.0 / 6.0) * M * d_16 * d_82 * dv_184 * dv_216 * dv_280 +
      (1.0 / 2.0) * M * d_26 * d_31 * d_37 * dv_164 * dv_180 * dv_293 * dv_72 *
          dv_89 * r0 -
      dv_175 * (d_125 * dv_58 + dv_290) - dv_280 * dv_281 - dv_280 * dv_291 -
      dv_289 * (2 * d_125 * d_22 * d_41 * dv_65 +
                4.0 * d_22 * d_41 * dv_283 * dv_58 - dv_288) -
      dv_328 *
          (d_10 * dv_226 * dv_295 + d_27 * dv_301 * dv_303 -
           dv_198 * dv_236 * dv_243 * dv_312 - dv_295 * dv_296 +
           dv_297 * dv_299 - dv_300 * dv_311 - dv_301 * dv_302 +
           dv_304 * dv_306 + dv_304 * dv_313 + dv_307 * dv_308 -
           dv_307 * dv_315 - dv_309 * dv_310 - dv_309 * dv_314 +
           dv_327 *
               (dv_199 *
                    (d_109 *
                         (d_128 * dv_319 +
                          d_14 * (Dx * d_72 * (-dv_116 + dv_15 - dv_261) +
                                  Dx * d_73 * dv_321 +
                                  30.0 * Dy * d_2 * d_70 * dv_0 -
                                  d_4 * dv_6 * (22 * dv_0 - dv_121 + dv_122) -
                                  d_70 * dv_11 * dv_257 -
                                  30.0 * d_71 * dv_0 * dv_54 - d_71 * dv_255) +
                          dv_106 * (-4 * dv_269 + 7.0 * dv_270 + 3.0 * dv_6) +
                          dv_294 * dv_320) +
                     d_126 * dv_322 + dv_317 * dv_318 +
                     dv_326 * (4 * d_1 * d_10 * d_13 * d_40 * dv_126 * dv_31 -
                               d_1 * dv_324 - d_1 * dv_325 - d_128 * dv_323 -
                               d_2 * dv_132 - d_2 * dv_262 +
                               2.0 * d_23 * d_40 * d_46 * dv_126 * dv_286 -
                               dv_263 * dv_317 -
                               dv_265 * (-3 * dv_269 + 4.0 * dv_270 + dv_6))) +
                dv_212 * dv_309 - dv_312 * dv_316));
  get<1>(get<::Tags::deriv<CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                           Frame::Inertial>>(*result)) =
      (1.0 / 6.0) * M * d_16 * d_82 * dv_184 * dv_216 * dv_333 +
      (1.0 / 2.0) * M * d_26 * d_31 * d_37 * dv_164 * dv_180 * dv_341 * dv_72 *
          dv_89 * r0 -
      dv_175 * (-d_129 * dv_58 + dv_339) - dv_281 * dv_333 -
      dv_289 * (-d_129 * d_48 * dv_66 + 4.0 * d_22 * d_41 * dv_276 * dv_58 -
                dv_287 * dv_336 - dv_337) -
      dv_291 * dv_333 -
      dv_328 *
          (12 * M * d_107 * d_27 * d_34 * dv_194 * r0 * dv_200 *
               (dv_195 * dv_198 * dv_201 * dv_348 + dv_196 * dv_198 * dv_344 +
                dv_196 * dv_201 * dv_346) +
           4.0 * d_10 * d_103 * d_27 * d_41 * d_69 * dv_188 * dv_191 * dv_194 *
               dv_344 +
           8.0 * d_10 * d_103 * d_41 * d_54 * dv_190 * dv_191 * dv_342 +
           2.0 * d_22 * d_27 * dv_200 * dv_201 * dv_206 *
               (dv_199 *
                    (d_109 *
                         (d_130 * dv_319 +
                          d_14 * (Dy * d_73 * (-dv_114 + dv_20 - dv_261) +
                                  Dy * dv_352 +
                                  d_5 * dv_4 *
                                      (68 * dv_0 - 22.0 * dv_1 - dv_122) +
                                  d_70 * dv_256 - d_70 * dv_350 * dv_53 -
                                  dv_254 * dv_351 + dv_350 * dv_351) +
                          d_2 * dv_31 * dv_320 +
                          dv_106 * (-4 * dv_273 + 7.0 * dv_275 + 3.0 * dv_4)) -
                     d_129 * dv_322 + dv_318 * dv_349 +
                     dv_326 * (2 * d_1 * d_10 * d_13 * d_40 * dv_32 +
                               d_1 * d_23 * d_40 * d_46 * dv_83 +
                               4.0 * d_10 * d_13 * d_2 * d_40 * dv_126 * dv_31 -
                               d_130 * dv_323 - d_2 * dv_324 - d_2 * dv_325 +
                               2.0 * d_23 * d_40 * d_46 * dv_126 * dv_336 -
                               dv_263 * dv_349 -
                               dv_265 * (-3 * dv_273 + 4.0 * dv_275 + dv_4))) +
                dv_212 * dv_347 + dv_316 * dv_348) +
           2.0 * d_22 * d_27 * dv_200 * dv_206 * dv_213 * dv_344 -
           dv_296 * dv_342 - dv_299 * dv_343 - dv_310 * dv_347 -
           dv_311 * dv_344 - dv_314 * dv_347 - dv_315 * dv_346 -
           dv_344 * dv_345);
  get<2>(get<::Tags::deriv<CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                           Frame::Inertial>>(*result)) =
      Dz *
      ((1.0 / 6.0) * M * d_134 * d_16 * d_82 * dv_184 * dv_216 +
       (1.0 / 2.0) * M * d_135 * d_26 * d_31 * d_37 * dv_164 * dv_180 * dv_72 *
           dv_89 * r0 -
       d_134 * dv_291 - 2.0 * d_52 * d_96 * dv_166 * (dv_51 - dv_59) -
       d_94 * dv_144 * (d_131 + d_132 - d_4 * d_63 - d_5 * d_63 + d_93) -
       dv_175 -
       dv_328 *
           (d_116 * d_134 * dv_303 + d_134 * dv_200 * dv_306 +
            d_134 * dv_200 * dv_313 - d_134 * dv_311 - d_134 * dv_345 +
            d_136 * dv_186 - d_136 * dv_192 + d_137 * dv_308 - d_137 * dv_315 +
            48.0 * d_96 * dv_244 * dv_353 - dv_310 - dv_314 +
            dv_327 *
                (d_67 * d_96 * dv_353 * dv_89 +
                 4.0 * dv_199 *
                     (2 * d_110 * d_79 * d_96 * dv_84 *
                          (d_119 * dv_128 + d_76 * dv_56 + 6.0 * dv_264) -
                      d_24 * d_66 * d_96 * dv_249 +
                      d_69 * d_78 *
                          (-d_96 * dv_105 +
                           r0 * (d_73 * dv_321 +
                                 d_74 * (-7 * dv_0 - 7.0 * dv_1 + 8.0 * dv_17) -
                                 dv_108 - dv_110 + dv_352))) +
                 dv_212)));
}
}  // namespace CurvedScalarWave::Worldtube
