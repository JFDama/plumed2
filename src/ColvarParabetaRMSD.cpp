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
#include "ActionRegister.h"
#include "PlumedMain.h"
#include "Atoms.h"

namespace PLMD {

//+PLUMEDOC COLVAR PARABETARMSD
/*
Probe the parallel beta sheet content of your protein structure.

Two protein segments containing three continguous residues can form a parallel beta sheet. 
Although if the two segments are part of the same protein chain they must be separated by 
a minimum of 3 residues to make room for the turn. This colvar thus generates the set of 
all possible six residue sections that could conceivably form a parallel beta sheet 
and calculates the RMSD distance between the configuration in which the residues find themselves
and an idealized parallel beta sheet structure. These distances can be calculated by either 
aligning the instantaneous structure with the reference structure and measuring each
atomic displacement or by calculating differences between the set of interatomic
distances in the reference and instantaneous structures. 

This colvar is based on the following reference \cite pietrucci09jctc.  The authors of 
this paper use the set of distances from the parallel beta sheet configurations to measure 
the number of segments that have an parallel beta sheet configuration. To do something 
similar using this implementation you must use the LESS_THAN keyword. Furthermore, 
based on reference \cite pietrucci09jctc we would recommend using the following
switching function definition (RATIONAL R_0=0.08 NN=8 MM=12) when your input file
is in units of nm. 

Please be aware that for codes like gromacs you must ensure that plumed 
reconstructs the chains involved in your CV when you calculate this CV using
anthing other than TYPE=DRMSD.  For more details as to how to do this see \ref WHOLEMOLECULES.

\par Examples

The following input calculates the number of six residue segments of 
protein that are in an parallel beta sheet configuration.

\verbatim
MOLINFO STRUCTURE=helix.pdb
PARABETARMSD BACKBONE=all TYPE=DRMSD LESS_THAN=(RATIONAL R_0=0.08 NN=8 MM=12) LABEL=a
\endverbatim
(see also \ref MOLINFO)

\par Examples

(see also \ref MOLINFO)

*/
//+ENDPLUMEDOC

class ColvarParabetaRMSD : public MultiColvarSecondaryStructureRMSD {
private:
  double s_cutoff;
public:
  static void registerKeywords( Keywords& keys );
  ColvarParabetaRMSD(const ActionOptions&);
  bool contributionIsSmall( std::vector<Vector>& pos );
  bool isPossibleToSkip();
}; 

PLUMED_REGISTER_ACTION(ColvarParabetaRMSD,"PARABETARMSD")

void ColvarParabetaRMSD::registerKeywords( Keywords& keys ){
  MultiColvarSecondaryStructureRMSD::registerKeywords( keys );
  keys.add("compulsory","STYLE","all","Parallel beta sheets can either form in a single chain or from a pair of chains. If STYLE=all all "
                                      "chain configuration with the appropriate geometry are counted.  If STYLE=inter "
                                      "only sheet-like configurations involving two chains are counted, while if STYLE=intra "
                                      "only sheet-like configurations involving a single chain are counted");
  keys.add("optional","STRANDS_CUTOFF","If in a segment of protein the two strands are further apart then the calculation "
                                       "of the actual RMSD is skipped as the structure is very far from being beta-sheet like. "
                                       "This keyword speeds up the calculation enormously when you are using the LESS_THAN option. "
                                       "However, if you are using some other option, then this cannot be used");
}

ColvarParabetaRMSD::ColvarParabetaRMSD(const ActionOptions&ao):
Action(ao),
MultiColvarSecondaryStructureRMSD(ao),
s_cutoff(0)
{
  // read in the backbone atoms
  std::vector<std::string> backnames(5); std::vector<unsigned> chains;
  backnames[0]="N"; backnames[1]="CA"; backnames[2]="CB"; backnames[3]="C"; backnames[4]="O";
  readBackboneAtoms( backnames, chains );

  bool intra_chain, inter_chain; 
  std::string style; parse("STYLE",style);
  if( style=="all" ){ 
      intra_chain=true; inter_chain=true;
  } else if( style=="inter"){
      intra_chain=false; inter_chain=true;
  } else if( style=="intra"){
      intra_chain=true; inter_chain=false;
  } else {
      error( style + " is not a valid directive for the STYLE keyword");
  }

  parse("STRANDS_CUTOFF",s_cutoff);
  if( s_cutoff>0) log.printf("  ignoring contributions from strands that are more than %f apart\n",s_cutoff);

  // This constructs all conceivable sections of antibeta sheet in the backbone of the chains
  if( intra_chain ){
    unsigned nres, nprevious=0; std::vector<unsigned> nlist(30);
    for(unsigned i=0;i<chains.size();++i){
       if( chains[i]<40 ) error("segment of backbone is not long enough to form an antiparallel beta hairpin. Each backbone fragment must contain a minimum of 8 residues");
       // Loop over all possible triples in each 8 residue segment of protein
       nres=chains[i]/5; plumed_assert( chains[i]%5==0 );
       for(unsigned ires=0;ires<nres-8;ires++){
           for(unsigned jres=ires+6;jres<nres-2;jres++){
               for(unsigned k=0;k<15;++k){
                  nlist[k]=nprevious + ires*5+k;
                  nlist[k+15]=nprevious + jres*5+k;
               }
               addColvar( nlist );
           }
       }
       nprevious+=chains[i];
    }
  }
  // This constructs all conceivable sections of antibeta sheet that form between chains
  if( inter_chain ){
      if( chains.size()==1 && style!="all" ) error("there is only one chain defined so cannot use inter_chain option");
      unsigned iprev,jprev,inres,jnres; std::vector<unsigned> nlist(30);
      for(unsigned ichain=1;ichain<chains.size();++ichain){
         iprev=0; for(unsigned i=0;i<ichain;++i) iprev+=chains[i];
         inres=chains[ichain]/5; plumed_assert( chains[ichain]%5==0 ); 
         for(unsigned ires=0;ires<inres-2;++ires){
            for(unsigned jchain=0;jchain<ichain;++jchain){
                jprev=0; for(unsigned i=0;i<jchain;++i) jprev+=chains[i];
                jnres=chains[jchain]/5; plumed_assert( chains[jchain]%5==0 );
                for(unsigned jres=0;jres<jnres-2;++jres){
                    for(unsigned k=0;k<15;++k){
                       nlist[k]=iprev + ires*5+k;
                       nlist[k+15]=jprev + jres*5+k;
                    } 
                    addColvar( nlist );
                }
            }
         }
      } 
  }

  // Build the reference structure ( in angstroms )
  std::vector<Vector> reference(30);
  reference[0]=Vector( 1.244, -4.620, -2.127); // N    i
  reference[1]=Vector(-0.016, -4.500, -1.395); // CA
  reference[2]=Vector( 0.105, -5.089,  0.024); // CB
  reference[3]=Vector(-0.287, -3.000, -1.301); // C
  reference[4]=Vector( 0.550, -2.245, -0.822); // O
  reference[5]=Vector(-1.445, -2.551, -1.779); // N    i+1
  reference[6]=Vector(-1.752, -1.130, -1.677); // CA
  reference[7]=Vector(-2.113, -0.550, -3.059); // CB
  reference[8]=Vector(-2.906, -0.961, -0.689); // C
  reference[9]=Vector(-3.867, -1.738, -0.695); // O
  reference[10]=Vector(-2.774,  0.034,  0.190); // N    i+2
  reference[11]=Vector(-3.788,  0.331,  1.201); // CA
  reference[12]=Vector(-3.188,  0.300,  2.624); // CB
  reference[13]=Vector(-4.294,  1.743,  0.937); // C
  reference[14]=Vector(-3.503,  2.671,  0.821); // O
  reference[15]=Vector( 4.746, -2.363,  0.188); // N    j
  reference[16]=Vector( 3.427, -1.839,  0.545); // CA
  reference[17]=Vector( 3.135, -1.958,  2.074); // CB
  reference[18]=Vector( 3.346, -0.365,  0.181); // C
  reference[19]=Vector( 4.237,  0.412,  0.521); // O
  reference[20]=Vector( 2.261,  0.013, -0.487); // N    j+1
  reference[21]=Vector( 2.024,  1.401, -0.875); // CA
  reference[22]=Vector( 1.489,  1.514, -2.313); // CB
  reference[23]=Vector( 0.914,  1.902,  0.044); // C
  reference[24]=Vector(-0.173,  1.330,  0.052); // O
  reference[25]=Vector( 1.202,  2.940,  0.828); // N    j+2
  reference[26]=Vector( 0.190,  3.507,  1.718); // CA
  reference[27]=Vector( 0.772,  3.801,  3.104); // CB
  reference[28]=Vector(-0.229,  4.791,  1.038); // C
  reference[29]=Vector( 0.523,  5.771,  0.996); // O
  // Store the secondary structure ( last number makes sure we convert to internal units nm )
  setSecondaryStructure( reference, 0.17/atoms.getUnits().length, 0.1/atoms.getUnits().length ); 

  reference[0]=Vector(-1.439, -5.122, -1.144); // N    i
  reference[1]=Vector(-0.816, -3.803, -1.013); // CA
  reference[2]=Vector( 0.099, -3.509, -2.206); // CB
  reference[3]=Vector(-1.928, -2.770, -0.952); // C
  reference[4]=Vector(-2.991, -2.970, -1.551); // O
  reference[5]=Vector(-1.698, -1.687, -0.215); // N    i+1
  reference[6]=Vector(-2.681, -0.613, -0.143); // CA
  reference[7]=Vector(-3.323, -0.477,  1.267); // CB
  reference[8]=Vector(-1.984,  0.681, -0.574); // C
  reference[9]=Vector(-0.807,  0.921, -0.273); // O
  reference[10]=Vector(-2.716,  1.492, -1.329); // N    i+2
  reference[11]=Vector(-2.196,  2.731, -1.883); // CA
  reference[12]=Vector(-2.263,  2.692, -3.418); // CB
  reference[13]=Vector(-2.989,  3.949, -1.433); // C
  reference[14]=Vector(-4.214,  3.989, -1.583); // O
  reference[15]=Vector( 2.464, -4.352,  2.149); // N    j
  reference[16]=Vector( 3.078, -3.170,  1.541); // CA
  reference[17]=Vector( 3.398, -3.415,  0.060); // CB
  reference[18]=Vector( 2.080, -2.021,  1.639); // C
  reference[19]=Vector( 0.938, -2.178,  1.225); // O
  reference[20]=Vector( 2.525, -0.886,  2.183); // N    j+1
  reference[21]=Vector( 1.692,  0.303,  2.346); // CA
  reference[22]=Vector( 1.541,  0.665,  3.842); // CB
  reference[23]=Vector( 2.420,  1.410,  1.608); // C
  reference[24]=Vector( 3.567,  1.733,  1.937); // O
  reference[25]=Vector( 1.758,  1.976,  0.600); // N    j+2
  reference[26]=Vector( 2.373,  2.987, -0.238); // CA
  reference[27]=Vector( 2.367,  2.527, -1.720); // CB
  reference[28]=Vector( 1.684,  4.331, -0.148); // C
  reference[29]=Vector( 0.486,  4.430, -0.415); // O
  // Store the secondary structure ( last number makes sure we convert to internal units nm )
  setSecondaryStructure( reference, 0.17/atoms.getUnits().length, 0.1/atoms.getUnits().length );
}

bool ColvarParabetaRMSD::isPossibleToSkip(){
  if(s_cutoff!=0) return true;
  return false;
}

bool ColvarParabetaRMSD::contributionIsSmall( std::vector<Vector>& pos ){
  if(s_cutoff==0) return false;

  Vector distance; distance=getSeparation( pos[6],pos[21] );  // This is the CA of the two residues at the centers of the two chains
  if( distance.modulo()>s_cutoff ) return true;

  // Align the two strands
  if( usingRMSD() ){
      Vector origin_old, origin_new; origin_old=pos[21];
      origin_new=pos[6]+distance;
      for(unsigned i=15;i<30;++i){
          pos[i]+=( origin_new - origin_old ); 
      }
  }   
  return false;
}

}
