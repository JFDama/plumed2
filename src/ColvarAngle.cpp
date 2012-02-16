#include "Colvar.h"
#include "ActionRegister.h"
#include "Angle.h"

#include <string>
#include <cmath>
#include <cassert>

using namespace std;

namespace PLMD{

//+PLUMEDOC COLVAR ANGLE
/**
Calculate the angle between three atoms.  Alternatively if four atoms appear in the atom
specification calculate the angle between the vector joining atoms 1 and 2 and that joineing
atoms 3 and 4.

\par Examples

This command tells plumed to calculate the angle between the vector connecting atom 1 to atom 2 and
the vector connecting atom 2 to atom 3
\verbatim
ANGLE ATOMS=1,2,3 
\endverbatim

This command tells plumed to calculate the angle between vector connecting atom 1 to atom 2 and
the vector connecting atom 3 to atom 4
\verbatim
ANGLE ATOMS=1,2,3,4 
\endverbatim

*/
//+ENDPLUMEDOC
   
class ColvarAngle : public Colvar {
  bool pbc;

public:
  ColvarAngle(const ActionOptions&);
// active methods:
  virtual void calculate();
  static void registerKeywords( Keywords& keys );
};

PLUMED_REGISTER_ACTION(ColvarAngle,"ANGLE")

void ColvarAngle::registerKeywords( Keywords& keys ){
  Colvar::registerKeywords(keys);
  keys.add("atoms","ATOMS","the list of atoms involved in this collective variable");
}

ColvarAngle::ColvarAngle(const ActionOptions&ao):
PLUMED_COLVAR_INIT(ao),
pbc(true)
{
  vector<AtomNumber> atoms;
  parseAtomList("ATOMS",atoms);
  bool nopbc=!pbc;
  parseFlag("NOPBC",nopbc);
  pbc=!nopbc;
  parseFlag("PBC",pbc);
  checkRead();

  if(atoms.size()==3){
    log.printf("  between atoms %d %d %d\n",atoms[0].serial(),atoms[1].serial(),atoms[2].serial());
    atoms.resize(4);
    atoms[3]=atoms[2];
    atoms[2]=atoms[1];
  }else if(atoms.size()==4){
    log.printf("  between lines %d-%d and %d-%d\n",atoms[0].serial(),atoms[1].serial(),atoms[2].serial(),atoms[3].serial());
  }else assert(0);

  if(pbc) log.printf("  using periodic boundary conditions\n");
  else    log.printf("  without periodic boundary conditions\n");

  addValueWithDerivatives(); setNotPeriodic();
  requestAtoms(atoms);
}

// calculator
void ColvarAngle::calculate(){

  Vector dij,dik;
  if(pbc){
    dij=pbcDistance(getPosition(2),getPosition(3));
    dik=pbcDistance(getPosition(1),getPosition(0));
  } else {
    dij=delta(getPosition(2),getPosition(3));
    dik=delta(getPosition(1),getPosition(0));
  }
  Vector ddij,ddik;
  Angle a;
  double angle=a.compute(dij,dik,ddij,ddik);
  setAtomsDerivatives(0,ddik);
  setAtomsDerivatives(1,-ddik);
  setAtomsDerivatives(2,-ddij);
  setAtomsDerivatives(3,ddij);
  setValue           (angle);
  setBoxDerivatives  (-(Tensor(dij,ddij)+Tensor(dik,ddik)));
}

}



