/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2011-2015 The plumed team
   (see the PEOPLE file at the root of the distribution for a list of names)

   See http://www.plumed-code.org for more information.

   This file is part of plumed, version 2.

   plumed is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   plumed is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with plumed.  If not, see <http://www.gnu.org/licenses/>.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */
#include "Bias.h"
#include "ActionRegister.h"
#include "../core/PlumedMain.h"


using namespace std;


namespace PLMD{
namespace bias{

//+PLUMEDOC BIAS RESTRAINT
/*
Adds harmonic and/or linear restraints on one or more variables.  

Either or both
of SLOPE and KAPPA must be present to specify the linear and harmonic force constants
respectively.  The resulting potential is given by: 
\f[
  \sum_i \frac{k_i}{2} (x_i-a_i)^2 + m_i*(x_i-a_i)
\f].

The number of components for any vector of force constants must be equal to the number
of arguments to the action.

Additional material and examples can be also found in the tutorial \ref belfast-4 

\par Examples
The following input tells plumed to restrain the distance between atoms 3 and 5
and the distance between atoms 2 and 4, at different equilibrium
values, and to print the energy of the restraint
\verbatim
DISTANCE ATOMS=3,5 LABEL=d1
DISTANCE ATOMS=2,4 LABEL=d2
RESTRAINT ARG=d1,d2 AT=1.0,1.5 KAPPA=150.0,150.0 LABEL=restraint
PRINT ARG=restraint.bias
\endverbatim
(See also \ref DISTANCE and \ref PRINT).

*/
//+ENDPLUMEDOC

class RegionSimStop : public Bias {
  std::vector<double> at;
  std::vector<double> tol;
  enum RegionType {kRectangular, kEllipsoidal};
  RegionType region_type;
public:
  explicit RegionSimStop(const ActionOptions&);
  void calculate();
  static void registerKeywords(Keywords& keys);
};

PLUMED_REGISTER_ACTION(RegionSimStop,"REGIONSIMSTOP")

void RegionSimStop::registerKeywords(Keywords& keys){
   Bias::registerKeywords(keys);
   keys.use("ARG");
   keys.add("compulsory","TOL","0.0","specifies that the simulation stop condition will trigger with this tolerance near the AT coordinate.");
   keys.add("compulsory","AT","the center position of the simulation stop trigger region");
   keys.add("optional","REGIONTYPE","the type of region to be used (rectangular or ellipsoidal)");
   componentsAreNotOptional(keys);
}

RegionSimStop::RegionSimStop(const ActionOptions&ao):
PLUMED_BIAS_INIT(ao),
at(getNumberOfArguments(),0.0),
tol(getNumberOfArguments(),0.0),
region_type(kEllipsoidal)
{
  parseVector("TOL",tol);
  parseVector("AT",at);
  std::string region_type_string = "ellipsoidal";
  parse("REGIONTYPE", region_type_string);
  if (region_type_string == "ellipsoidal") {
    region_type = kEllipsoidal;
  } else if (region_type_string == "rectangular") {
    region_type = kRectangular;
  } else {
    error("unrecognized REGIONTYPE; choose one of 'ellipsoidal' or 'rectangular'");
  }
  checkRead();

  log.printf("  at");
  for(unsigned i=0;i<at.size();i++) log.printf(" %f",at[i]);
  log.printf("\n");
  log.printf("  with tolerances in each direction of");
  for(unsigned i=0;i<tol.size();i++) log.printf(" %f",tol[i]);
  log.printf("\n");
  if (region_type == kRectangular) {
    log.printf("  using a rectangular region shape\n");
  } else if (region_type == kEllipsoidal) {
    log.printf("  using an ellipsoidal region shape\n");
  }

}

void RegionSimStop::calculate(){
  double sumsq = 0.0;
  bool is_in_rect = true;
  for(unsigned i = 0; i < getNumberOfArguments(); ++i){
    const double cv = abs(difference(i, at[i], getArgument(i))) / tol[i];
    if (cv > 1.0) {
      is_in_rect = false;
    }
    if (region_type == kEllipsoidal) {
      // Add up the violation if ellipsoidal, to check for
      // between-axes enclosure.
      sumsq += cv * cv;
    }
    setOutputForce(i, 0.0);
  }
  if (is_in_rect) {
    if (region_type == kRectangular) {
      // Stop the simulation.
      plumed.stop();
    } else if (region_type == kEllipsoidal && sumsq < 1.0) {
      // Stop the simulation.
      plumed.stop();
    }
  }
}

}


}
