#include "ActionAtomistic.h"
#include "PlumedMain.h"
#include "ActionSet.h"
#include <vector>
#include <string>
#include "ActionWithValue.h"
#include "Colvar.h"
#include "ActionWithVirtualAtom.h"
#include "PlumedException.h"
#include "Atoms.h"

using namespace std;
using namespace PLMD;

ActionAtomistic::~ActionAtomistic(){
// forget the pending request
  atoms.remove(this);
}

ActionAtomistic::ActionAtomistic(const ActionOptions&ao):
Action(ao),
lockRequestAtoms(false),
atoms(plumed.getAtoms())
{
  atoms.add(this);
}

void ActionAtomistic::registerKeywords( Keywords& keys ){
  (void) keys; // avoid warning
}


void ActionAtomistic::requestAtoms(const vector<AtomNumber> & a){
  plumed_massert(!lockRequestAtoms,"requested atom list can only be changed in the prepare() method");
  int nat=a.size();
  indexes=a;
  positions.resize(nat);
  forces.resize(nat);
  masses.resize(nat);
  charges.resize(nat);
  int n=atoms.positions.size();
  clearDependencies();
  unique.clear();
  for(unsigned i=0;i<indexes.size();i++){
    plumed_massert(indexes[i].index()<n,"atom out of range");
    if(atoms.isVirtualAtom(indexes[i])) addDependency(atoms.getVirtualAtomsAction(indexes[i]));
// only real atoms are requested to lower level Atoms class
    else unique.insert(indexes[i]);
  }

}

Vector ActionAtomistic::pbcDistance(const Vector &v1,const Vector &v2)const{
  return pbc.distance(v1,v2);
}

void ActionAtomistic::calculateNumericalDerivatives( ActionWithValue* a ){
  if(!a){
    a=dynamic_cast<ActionWithValue*>(this);
    plumed_massert(a,"only Actions with a value can be differentiated");
  }

  const int nval=a->getNumberOfComponents();
  const int natoms=getNumberOfAtoms();
  std::vector<Vector> value(nval*natoms);
  std::vector<Tensor> valuebox(nval);
  std::vector<Vector> savedPositions(natoms);
  const double delta=sqrt(epsilon);

  for(int i=0;i<natoms;i++) for(int k=0;k<3;k++){
    savedPositions[i][k]=positions[i][k];
    positions[i][k]=positions[i][k]+delta;
    a->calculate();
    positions[i][k]=savedPositions[i][k];
    for(unsigned j=0;j<nval;j++){
      value[j*natoms+i][k]=a->getOutputQuantity(j);
    }
  }
 for(int i=0;i<3;i++) for(int k=0;k<3;k++){
   double arg0=box(i,k);
   for(int j=0;j<natoms;j++) positions[j]=pbc.realToScaled(positions[j]);
   box(i,k)=box(i,k)+delta;
   pbc.setBox(box);
   for(int j=0;j<natoms;j++) positions[j]=pbc.scaledToReal(positions[j]);
   a->calculate();
   box(i,k)=arg0;
   pbc.setBox(box);
   for(int j=0;j<natoms;j++) positions[j]=savedPositions[j];
   for(unsigned j=0;j<nval;j++) valuebox[j](i,k)=a->getOutputQuantity(j);
 }

  a->calculate();
  a->clearDerivatives();
  for(unsigned j=0;j<nval;j++){
    Value* v=a->copyOutput(j);
    double ref=v->get();
    if(v->hasDerivatives()){
      for(int i=0;i<natoms;i++) for(int k=0;k<3;k++) {
        double d=(value[j*natoms+i][k]-ref)/delta;
        v->addDerivative(3*i+k,d);
      }
      Tensor virial;
      for(int i=0;i<3;i++) for(int k=0;k<3;k++)virial(i,k)= (valuebox[j](i,k)-ref)/delta;
// BE CAREFUL WITH NON ORTHOROMBIC CELL
      virial=-1.0*matmul(box.transpose(),virial.transpose());
      for(int i=0;i<3;i++) for(int k=0;k<3;k++) v->addDerivative(3*natoms+3*k+i,virial(i,k));
    }
  }
}

void ActionAtomistic::parseAtomList(const std::string&key, std::vector<AtomNumber> &t){
  parseAtomList(key,-1,t);
}

void ActionAtomistic::parseAtomList(const std::string&key,const int num, std::vector<AtomNumber> &t){
  plumed_massert( keywords.style(key,"atoms"), "keyword " + key + " should be registered as atoms");
  vector<string> strings;
  if( num<0 ){
      parseVector(key,strings);
      if(strings.empty()) return;
  } else {
      if ( !parseNumberedVector(key,num,strings) ) return;
  }

  Tools::interpretRanges(strings); t.resize(0);
  for(unsigned i=0;i<strings.size();++i){
   bool ok=false;
   AtomNumber atom;
   ok=Tools::convert(strings[i],atom); // this is converting strings to AtomNumbers
   if(ok) t.push_back(atom);
// here we check if the atom name is the name of a group
   if(!ok){
     if(atoms.groups.count(strings[i])){
       map<string,vector<AtomNumber> >::const_iterator m=atoms.groups.find(strings[i]);
       t.insert(t.end(),m->second.begin(),m->second.end());
       ok=true;
     }
   }
// here we check if the atom name is the name of an added virtual atom
   if(!ok){
     const ActionSet&actionSet(plumed.getActionSet());
     for(ActionSet::const_iterator a=actionSet.begin();a!=actionSet.end();++a){
       ActionWithVirtualAtom* c=dynamic_cast<ActionWithVirtualAtom*>(*a);
       if(c) if(c->getLabel()==strings[i]){
         ok=true;
         t.push_back(c->getIndex());
         break;
       }
     }
   }
   plumed_massert(ok,"it was not possible to interpret atom name " + strings[i]);
  }
} 


void ActionAtomistic::retrieveAtoms(){
  box=atoms.box;
  pbc.setBox(box);
  const vector<Vector> & p(atoms.positions);
  const vector<double> & c(atoms.charges);
  const vector<double> & m(atoms.masses);
  for(unsigned j=0;j<indexes.size();j++) positions[j]=p[indexes[j].index()];
  for(unsigned j=0;j<indexes.size();j++) charges[j]=c[indexes[j].index()];
  for(unsigned j=0;j<indexes.size();j++) masses[j]=m[indexes[j].index()];
  Colvar*cc=dynamic_cast<Colvar*>(this);
  if(cc && cc->checkIsEnergy()) energy=atoms.getEnergy();
}

void ActionAtomistic::applyForces(){
  vector<Vector>   & f(atoms.forces);
  Tensor           & v(atoms.virial);
  for(unsigned j=0;j<indexes.size();j++) f[indexes[j].index()]+=forces[j];
  v+=virial;
  atoms.forceOnEnergy+=forceOnEnergy;
}

void ActionAtomistic::readAndCalculate( const PDB& pdb ){
  for(unsigned j=0;j<indexes.size();j++){
      if( indexes[j].index()>pdb.size() ) error("there are not enough atoms in the input pdb file");
      if( pdb.getAtomNumbers()[j].index()!=j ) error("there are atoms missing in the pdb file");  
      positions[j]=pdb.getPositions()[indexes[j].index()];
  }
  for(unsigned j=0;j<indexes.size();j++) charges[j]=pdb.getBeta()[indexes[j].index()];
  for(unsigned j=0;j<indexes.size();j++) masses[j]=pdb.getOccupancy()[indexes[j].index()];
  prepare(); calculate();
}






