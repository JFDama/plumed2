/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2012 The plumed team
   (see the PEOPLE file at the root of the distribution for a list of names)

   See http://www.plumed-code.org for more information.

   This file is part of plumed, version 2.0.

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
#include "MultiColvarSecondaryStructureRMSD.h"
#include "PlumedMain.h"
#include "Atoms.h"

namespace PLMD {

void MultiColvarSecondaryStructureRMSD::registerKeywords( Keywords& keys ){
  MultiColvar::registerKeywords( keys );
  ActionWithDistribution::autoParallelize( keys );
  keys.add("atoms","RESIDUES","this command is used to specify the set of residues that could conceivably form part of the secondary structure. "
                              "It is possible to use residues numbers as the various chains and residues should have been identified else using an instance of the "
                              "\\ref MOLINFO action. If you wish to use all the residues from all the chains in your system you can do so by "
                              "specifying all. Alternatively, if you wish to use a subset of the residues you can specify the particular residues "
                              "you are interested in as a list of numbers. Please be aware that to form secondary structure elements your chain "
                              "must contain at least N residues, where N is dependent on the particular secondary structure you are interested in. "
                              "As such if you define portions of the chain with fewer than N residues the code will crash."); 
  keys.add("compulsory","TYPE","DRMSD","the manner in which RMSD alignment is performed. Should be OPTIMAL, SIMPLE or DRMSD.");
  keys.use("LESS_THAN"); keys.use("MIN"); keys.use("AVERAGE");
}

MultiColvarSecondaryStructureRMSD::MultiColvarSecondaryStructureRMSD(const ActionOptions&ao):
Action(ao),
MultiColvar(ao)
{
  parse("TYPE",alignType);
  log.printf("  distances from secondary structure elements are calculated using %s algorithm\n",alignType.c_str() );
  log<<"  Bibliography "<<plumed.cite("Pietrucci and Laio, J. Chem. Theory Comput. 5, 2197 (2009)"); log<<"\n";
}

MultiColvarSecondaryStructureRMSD::~MultiColvarSecondaryStructureRMSD(){
  for(unsigned i=0;i<secondary_rmsd.size();++i) delete secondary_rmsd[i]; 
  for(unsigned i=0;i<secondary_drmsd.size();++i) delete secondary_drmsd[i];
}

void MultiColvarSecondaryStructureRMSD::setSecondaryStructure( std::vector<Vector>& structure, double bondlength, double units ){

  // Convert into correct units
  for(unsigned i=0;i<structure.size();++i){
     structure[i][0]*=units; structure[i][1]*=units; structure[i][2]*=units;
  }

  if( secondary_drmsd.size()==0 && secondary_rmsd.size()==0 ){ 
     int natoms; readAtoms(natoms); requestDistribution();
  }

  // If we are in natural units get conversion factor from nm into natural length units
  if( plumed.getAtoms().usingNaturalUnits() ){
      error("cannot use this collective variable when using natural units");
  }

  // Set the reference structure
  new_deriv.resize( structure.size() );
  if( alignType=="DRMSD" ){
    secondary_drmsd.push_back( new DRMSD() ); 
    secondary_drmsd[secondary_drmsd.size()-1]->setReference( structure, bondlength );
  } else {
    std::vector<double> align( structure.size(), 1.0 );
    secondary_rmsd.push_back( new RMSD(log) );
    secondary_rmsd[secondary_rmsd.size()-1]->setType( alignType );
    secondary_rmsd[secondary_rmsd.size()-1]->setReference( structure );
    secondary_rmsd[secondary_rmsd.size()-1]->setAlign( align ); 
    secondary_rmsd[secondary_rmsd.size()-1]->setDisplace( align );
  }
}

double MultiColvarSecondaryStructureRMSD::compute( const unsigned& j, const std::vector<Vector>& pos, std::vector<Vector>& deriv, Tensor& virial ){
  double r,nr; Tensor new_virial;

  if( secondary_drmsd.size()>0 ){
    r=secondary_drmsd[0]->calculate( pos, getPbc(), deriv, virial ); 
    for(unsigned i=1;i<secondary_drmsd.size();++i){
        if( usesPbc() ) nr=secondary_drmsd[i]->calculate( pos, getPbc(), new_deriv, new_virial );
        else nr=secondary_drmsd[i]->calculate( pos, getPbc(), new_deriv, new_virial );
        if(nr<r){
           r=nr;
           for(unsigned i=0;i<new_deriv.size();++i) deriv[i]=new_deriv[i];
           virial=new_virial; 
        }
    }
  } else {
    r=secondary_rmsd[0]->calculate( pos, deriv );
    for(unsigned i=1;i<secondary_rmsd.size();++i){
        nr=secondary_rmsd[i]->calculate( pos, new_deriv );
        if(nr<r){
           r=nr;
           for(unsigned i=0;i<new_deriv.size();++i) deriv[i]=new_deriv[i];
        }
    } 
    for(unsigned i=0;i<deriv.size();i++) virial=virial+(-1.0*Tensor(pos[i],deriv[i]));
  }
  return r;
}

unsigned MultiColvarSecondaryStructureRMSD::getNumberOfFieldDerivatives(){
  plumed_massert(0,"Fields are not allowed for secondary structure variables");
}

bool MultiColvarSecondaryStructureRMSD::usingRMSD() const {
  if( secondary_rmsd.size()>0 ) return true;
  else return false;
}

}
