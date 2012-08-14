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
#include "Bias.h"
#include "ActionRegister.h"
#include "Grid.h"
#include "PlumedMain.h"
#include "Atoms.h"
#include "PlumedException.h"
#include "FlexibleBin.h"
#include "Matrix.h"
#include "Random.h"
#include "PlumedFile.h"

#define DP2CUTOFF 6.25

using namespace std;


namespace PLMD{

//+PLUMEDOC BIAS METAD 
/*
Used to performed MetaDynamics on one or more collective variables.

In a metadynamics simulations a history dependent bias composed of 
intermittently added Gaussian functions is added to the potential \cite metad.

\f[
V(\vec{s},t) = \sum_{ k \tau < t} W(k \tau)
\exp\left(
-\sum_{i=1}^{d} \frac{(s_i-s_i^{(0)}(k \tau))^2}{2\sigma_i^2}
\right).
\f]

This potential forces the system away from the kinetic traps in the potential energy surface
and out into the unexplored parts of the energy landscape. Information on the Gaussian
functions from which this potential is composed is output to a file called HILLS, which 
is used both the restart the calculation and to reconstruct the free energy as a function of the CVs. 
The free energy can be reconstructed from a metadynamics calculation because the final bias is given
by: 

\f[
V(\vec{s}) = -F(\vec(s))
\f]

During post processing the free energy can be calculated in this way using the \subpage sum_hills
utility.

In the simplest possible implementation of a metadynamics calculation the expense of a metadynamics 
calculation increases with the length of the simulation as one has to, at every step, evaluate 
the values of a larger and larger number of Gaussians. To avoid this issue you can in plumed 2.0 
store the bias on a grid.  This approach is similar to that proposed in \cite babi+08jcp but has the 
advantage that the grid spacing is independent on the Gaussian width.

Another option that is available in plumed 2.0 is well-tempered metadynamics \cite Barducci:2008. In this
varient of metadynamics the heights of the Gaussian hills are rescaled at each step so the bias is now
given by:

\f[
V({s},t)= \sum_{t'=0,\tau_G,2\tau_G,\dots}^{t'<t} W e^{-V({s}({q}(t'),t')/\Delta T} \exp\left(
-\sum_{i=1}^{d} \frac{(s_i({q})-s_i({q}(t'))^2}{2\sigma_i^2}
\right),
\f]

This method ensures that the bias converges more smoothly. 

\par Examples
The following input is for a standard metadynamics calculation using as
collective variables the distance between atoms 3 and 5
and the distance between atoms 2 and 4. The value of the CVs and
the metadynamics bias potential are written to the COLVAR file every 100 steps.
\verbatim
DISTANCE ATOMS=3,5 LABEL=d1
DISTANCE ATOMS=2,4 LABEL=d2
METAD ARG=d1,d2 SIGMA=0.2,0.2 HEIGHT=0.3 PACE=500 LABEL=restraint
PRINT ARG=d1,d2,restraint.bias STRIDE=100  FILE=COLVAR
\endverbatim
(See also \ref DISTANCE \ref PRINT).

*/
//+ENDPLUMEDOC

class BiasMetaD : public Bias{

private:
//  struct Gaussian {
//   vector<double> center;
//   vector<double> sigma;
//   double height;
//   bool   multivariate; // this is required to discriminate the one dimensional case 
//   vector<double> invsigma;
//   Gaussian(const vector<double> & center,const vector<double> & sigma,double height, bool multivariate ):
//     center(center),sigma(sigma),height(height),multivariate(multivariate),invsigma(sigma){
//       for(unsigned i=0;i<invsigma.size();++i)abs(invsigma[i])>1.e-20?invsigma[i]=1.0/invsigma[i]:0.; // to avoid troubles from zero element in flexible hills
//     }
//  };
  vector<double> sigma0_;
  vector<Kernel> hills_;
  PlumedOFile hillsOfile_;
  Grid* BiasGrid_;
  FILE* gridfile_;
  double height0_;
  double biasf_;
  double temp_;
  double* dp_;
  int stride_;
  int wgridstride_; 
  bool welltemp_;
  bool restart_;
  bool grid_;
  int adaptive_;
  FlexibleBin *flexbin;
  std::string kerneltype;
  std::vector<std::string> argument_names;

/* PUT BACK IN  
  void   readGaussians(PlumedIFile&);
*/
//  void   writeGaussian(const Gaussian&,PlumedOFile&);
//  void   addGaussian(const Gaussian&);
  double getHeight(const vector<double>&);
//  double getBiasAndDerivatives(const vector<double>&,double* der=NULL);
//  double evaluateGaussian(const vector<double>&, const Gaussian&,double* der=NULL);
//  void   finiteDifferenceGaussian(const vector<double>&, const Gaussian&);
//  vector<unsigned> getGaussianSupport(const Gaussian&);


public:
  BiasMetaD(const ActionOptions&);
  ~BiasMetaD();
  void calculate();
  void update();
  static void registerKeywords(Keywords& keys);
  bool checkNeedsGradients()const{if(adaptive_==FlexibleBin::geometry){return true;}else{return false;}};
};

PLUMED_REGISTER_ACTION(BiasMetaD,"METAD")

void BiasMetaD::registerKeywords(Keywords& keys){
  Bias::registerKeywords(keys);
  keys.use("ARG");
  keys.add("compulsory","SIGMA","the widths of the Gaussian hills");
  keys.add("compulsory","HEIGHT","the heights of the Gaussian hills");
  keys.add("compulsory","PACE","the frequency for hill addition");
  keys.add("compulsory","FILE","HILLS","a file in which the list of added hills is stored");
  keys.addFlag("RESTART",false,"restart the calculation from a previous metadynamics calculation.");
  keys.add("optional","BIASFACTOR","use well tempered metadynamics and use this biasfactor.  Please note you must also specify temp");
  keys.add("optional","TEMP","the system temperature - this is only needed if you are doing well-tempered metadynamics");
  keys.add("optional","GRID_MIN","the lower bounds for the grid");
  keys.add("optional","GRID_MAX","the upper bounds for the grid");
  keys.add("optional","GRID_BIN","the number of bins for the grid");
  keys.addFlag("GRID_SPARSE",false,"use a sparse grid to store hills");
  keys.addFlag("GRID_NOSPLINE",false,"don't use spline interpolation with grids");
  keys.add("optional","GRID_WSTRIDE","write the grid to a file every N steps");
  keys.add("optional","GRID_WFILE","the file on which to write the grid");
  keys.add("optional","ADAPTIVE","use a geometric (=GEOM) or diffusion (=DIFF) based hills width scheme. Sigma is one number that has distance or time dimensions");
  keys.add("optional","KERNEL","use a non-gaussian kernel function");
}

BiasMetaD::~BiasMetaD(){
  if(BiasGrid_) delete BiasGrid_;
  hillsOfile_.close();
  if(gridfile_) fclose(gridfile_);
  delete [] dp_;
}

BiasMetaD::BiasMetaD(const ActionOptions& ao):
PLUMED_BIAS_INIT(ao),
BiasGrid_(NULL),
gridfile_(NULL),
height0_(0.0),
biasf_(1.0),
temp_(0.0),
dp_(NULL),
stride_(0),
wgridstride_(0),
welltemp_(false),
restart_(false),
grid_(false),
adaptive_(FlexibleBin::none)
{
  // parse the flexible hills
  string adaptiveoption;
  adaptiveoption="NONE";
  parse("ADAPTIVE",adaptiveoption);
  if (adaptiveoption=="GEOM"){
		  log.printf("  Uses Geometry-based hills width: sigma must be in distance units and need only one sigma\n");
		  adaptive_=FlexibleBin::geometry;	
  }else if (adaptiveoption=="DIFF"){
		  log.printf("  Uses Diffusion-based hills width: sigma must be in time units and need only one sigma\n");
		  adaptive_=FlexibleBin::diffusion;	
  }else if (adaptiveoption=="NONE"){
		  adaptive_=FlexibleBin::none;	
  }else{
		  plumed_merror("I do not know this type of adaptive scheme");	
  }
  // parse the sigma
  parseVector("SIGMA",sigma0_);

  // if you use normal sigma you need one sigma per argument 
  if (adaptive_==FlexibleBin::none){
 	 plumed_assert(sigma0_.size()==getNumberOfArguments());
  }else{
  // if you use flexible hills you need one sigma  
         if(sigma0_.size()!=1){
	         plumed_merror("If you choose ADAPTIVE you need only one sigma according to your choice of the type");
         } 
  	 flexbin=new FlexibleBin(adaptive_,this,sigma0_[0]); 
  }
  parse("HEIGHT",height0_);
  plumed_assert(height0_>0.0);
  parse("PACE",stride_);
  plumed_assert(stride_>0);
  string hillsfname="HILLS";
  parse("FILE",hillsfname);
  parseFlag("RESTART",restart_);
  parse("BIASFACTOR",biasf_);
  plumed_assert(biasf_>=1.0);
  parse("TEMP",temp_);
  if(biasf_>1.0){
   plumed_assert(temp_>0.0);
   welltemp_=true;
  }

  vector<double> gmin(getNumberOfArguments());
  parseVector("GRID_MIN",gmin);
  plumed_assert(gmin.size()==getNumberOfArguments() || gmin.size()==0);
  vector<double> gmax(getNumberOfArguments());
  parseVector("GRID_MAX",gmax);
  plumed_assert(gmax.size()==getNumberOfArguments() || gmax.size()==0);
  vector<unsigned> gbin(getNumberOfArguments());
  parseVector("GRID_BIN",gbin);
  plumed_assert(gbin.size()==getNumberOfArguments() || gbin.size()==0);
  plumed_assert(gmin.size()==gmax.size() && gmin.size()==gbin.size());
  bool sparsegrid=false;
  parseFlag("GRID_SPARSE",sparsegrid);
  bool nospline=false;
  parseFlag("GRID_NOSPLINE",nospline);
  bool spline=!nospline;
  if(gbin.size()>0){grid_=true;}
  parse("GRID_WSTRIDE",wgridstride_);
  string gridfname;
  parse("GRID_WFILE",gridfname); 

  if(grid_&&gridfname.length()>0){plumed_assert(wgridstride_>0);}
  if(grid_&&wgridstride_>0){plumed_assert(gridfname.length()>0);}

  kerneltype="";
  if(grid_) parse("KERNEL",kerneltype);
  if(kerneltype.length()==0) kerneltype="gaussian";
  checkRead();

  log.printf("  Gaussian width ");
  if (adaptive_==FlexibleBin::diffusion)log.printf(" (Note: The units of sigma are in time) ");
  if (adaptive_==FlexibleBin::geometry)log.printf(" (Note: The units of sigma are in dist units) ");
  for(unsigned i=0;i<sigma0_.size();++i) log.printf(" %f",sigma0_[i]);
  log.printf("  Gaussian height %f\n",height0_);
  log.printf("  Gaussian deposition pace %d\n",stride_); 
  log.printf("  Gaussian file %s\n",hillsfname.c_str());
  if(welltemp_){log.printf("  Well-Tempered Bias Factor %f\n",biasf_);}
  if(grid_){
   log.printf("  Grid min");
   for(unsigned i=0;i<gmin.size();++i) log.printf(" %f",gmin[i]);
   log.printf("\n");
   log.printf("  Grid max");
   for(unsigned i=0;i<gmax.size();++i) log.printf(" %f",gmax[i]);
   log.printf("\n");
   log.printf("  Grid bin");
   for(unsigned i=0;i<gbin.size();++i) log.printf(" %d",gbin[i]);
   log.printf("\n");
   if(spline){log.printf("  Grid uses spline interpolation\n");}
   if(sparsegrid){log.printf("  Grid uses sparse grid\n");}
   if(wgridstride_>0){log.printf("  Grid is written on file %s with stride %d\n",gridfname.c_str(),wgridstride_);} 
 }
  
  addComponent("bias");

// for performance
   dp_ = new double[getNumberOfArguments()];

// initializing grid
  if(grid_){
   vector<bool> pbc;
   for(unsigned i=0;i<getNumberOfArguments();++i){
    pbc.push_back(getPntrToArgument(i)->isPeriodic());
// if periodic, use CV domain for grid boundaries
    if(pbc[i]){
     double dmin,dmax;
     getPntrToArgument(i)->getDomain(dmin,dmax);
     gmin[i]=dmin;
     gmax[i]=dmax;
    }
   }
   if(!sparsegrid){BiasGrid_=new Grid(gmin,gmax,gbin,pbc,spline,true);}
   else{BiasGrid_=new SparseGrid(gmin,gmax,gbin,pbc,spline,true);}
// open file for grid writing 
   if(wgridstride_>0){gridfile_=fopen(gridfname.c_str(),"w");}
  }

// restarting from HILLS file
  if(restart_){
   log.printf("  Restarting from %s:",hillsfname.c_str());
   PlumedIFile ifile;
   ifile.link(*this);
   ifile.open(hillsfname,"r");
/* PUT BACK IN
   readGaussians(ifile);
*/
   ifile.close();
   hillsOfile_.link(*this);
   hillsOfile_.open(hillsfname,"aw");
  }else{
   hillsOfile_.link(*this);
   hillsOfile_.open(hillsfname,"w");
  } 
  hillsOfile_.addConstantField("multivariate");

  // Check the kernel is OK
  std::vector<double> pp( getNumberOfArguments() ), thissig;
  if(adaptive_!=FlexibleBin::none) thissig.resize( getNumberOfArguments()*(getNumberOfArguments()+1) / 2 ); 
  else thissig.resize( getNumberOfArguments() );
  
  Kernel kernel( pp, thissig, kerneltype, 1.0, false );
  argument_names.resize( getNumberOfArguments() );
  for(unsigned i=0;i<getNumberOfArguments();++i) argument_names[i]=getPntrToArgument(i)->getName(); 

  log<<"  Bibliography "<<plumed.cite("Laio and Parrinello, PNAS 99, 12562 (2002)");
  if(welltemp_) log<<plumed.cite(
    "Barducci, Bussi, and Parrinello, Phys. Rev. Lett. 100, 020603 (2008)");
  log<<"\n";

}

/*
void BiasMetaD::readGaussians(PlumedIFile&ifile)
{
 unsigned ncv=getNumberOfArguments();
 double dummy;
 bool multivariate=false;
 vector<double> center(ncv);
 vector<double> sigma(ncv);
 double height;
 int nhills=0;
 while(ifile.scanField("time",dummy)){
  for(unsigned i=0;i<ncv;++i)
    ifile.scanField(getPntrToArgument(i)->getName(),center[i]);
  // scan for multivariate label: record the actual file position so to eventually rewind 
    std::string sss;
    ifile.scanField("multivariate",sss);
    if(sss=="true") multivariate=true;
    else if(sss=="false") multivariate=false;
    else plumed_merror("cannot parse multivariate = "+ sss);
  if(multivariate){
        sigma.resize(ncv*(ncv+1)/2);
        Matrix<double> upper(ncv,ncv);
        Matrix<double> lower(ncv,ncv);
	for (unsigned i=0;i<ncv;i++){
              for (unsigned j=0;j<ncv-i;j++){
                      ifile.scanField("sigma_"+getPntrToArgument(j+i)->getName()+"_"+getPntrToArgument(j)->getName(),lower(j+i,j));
                      upper(j,j+i)=lower(j+i,j);
              }
         }
        Matrix<double> mymult(ncv,ncv);       
        Matrix<double> invmatrix(ncv,ncv);       
        mult(lower,upper,mymult);          
        // now invert and get the sigmas
        Invert(mymult,invmatrix);
        // put the sigmas in the usual order 
        unsigned k=0;
	for (unsigned i=0;i<ncv;i++){
		for (unsigned j=i;j<ncv;j++){
			sigma[k]=invmatrix(i,j);
			k++;
		}
	}
  }else{
  	for(unsigned i=0;i<ncv;++i)ifile.scanField("sigma_"+getPntrToArgument(i)->getName(),sigma[i]);
  }

  ifile.scanField("height",height);
  ifile.scanField("biasf",dummy);
  ifile.scanField();
  nhills++;
  if(welltemp_){height*=(biasf_-1.0)/biasf_;}
 
  if(!grid_){
     hills_.push_back( GaussianKernel( KernelOptions(center,sigma,height,false) ) ); 
  } else {
     Kernel* kernel=KernelRegister::create( kerneltype, KernelOptions(center, sigma, height, false), true );
     BiasGrid_->addKernel( kernel ); 
     delete kernel; 
  }
//  addGaussian( GaussianKernel( KernelOptions(center,sigma,height,true) ) );
 }     
 log.printf("  %d Gaussians read\n",nhills);
}
*/

/*
void BiasMetaD::writeGaussian(const Gaussian& hill, PlumedOFile&file){
  unsigned ncv=getNumberOfArguments();
  file.printField("time",getTimeStep()*getStep());
  for(unsigned i=0;i<ncv;++i){
    file.printField(getPntrToArgument(i)->getName(),hill.center[i]);
  }
  if(hill.multivariate){
    hillsOfile_.printField("multivariate","true");
         Matrix<double> mymatrix(ncv,ncv);
         unsigned k=0;
         for(unsigned i=0;i<ncv;i++){
                for(unsigned j=i;j<ncv;j++){
                        mymatrix(i,j)=mymatrix(j,i)=hill.sigma[k]; // recompose the full inverse matrix
                        k++;
                }
         }
         // invert it 
         Matrix<double> invmatrix(ncv,ncv);
         Invert(mymatrix,invmatrix);
         // enforce symmetry
         for(unsigned i=0;i<ncv;i++){
                for(unsigned j=i;j<ncv;j++){
                        invmatrix(i,j)=invmatrix(j,i);
                }
         }

         // do cholesky so to have a "sigma like" number
         Matrix<double> lower(ncv,ncv);
         cholesky(invmatrix,lower); // now this , in band form , is similar to the sigmas
         // loop in band form 
         for (unsigned i=0;i<ncv;i++){
              for (unsigned j=0;j<ncv-i;j++){
                      file.printField("sigma_"+getPntrToArgument(j+i)->getName()+"_"+getPntrToArgument(j)->getName(),lower(j+i,j));
              }
         }
  } else {
    hillsOfile_.printField("multivariate","false");
    for(unsigned i=0;i<ncv;++i)
      file.printField("sigma_"+getPntrToArgument(i)->getName(),hill.sigma[i]);
  }
  double height=hill.height;
  if(welltemp_){height*=biasf_/(biasf_-1.0);}
  file.printField("height",height).printField("biasf",biasf_);
  file.printField();
}
*/

/*
void BiasMetaD::addGaussian(const Gaussian& hill)
{
 if(!grid_){hills_.push_back(hill);} 
 else{
  unsigned ncv=getNumberOfArguments();
  vector<unsigned> nneighb=getGaussianSupport(hill);
  vector<unsigned> neighbors=BiasGrid_->getNeighbors(hill.center,nneighb);
  vector<double> der(ncv);
  vector<double> xx(ncv);
  for(unsigned i=0;i<neighbors.size();++i){
   unsigned ineigh=neighbors[i];
   for(unsigned j=0;j<ncv;++j){der[j]=0.0;}
   BiasGrid_->getPoint(ineigh,xx);   
   double bias=evaluateGaussian(xx,hill,&der[0]);
   BiasGrid_->addValueAndDerivatives(ineigh,bias,der);
  }
 }
}

vector<unsigned> BiasMetaD::getGaussianSupport(const Gaussian& hill)
{
 vector<unsigned> nneigh;
 // traditional or flexible hill? 
 if(hill.multivariate){
	unsigned ncv=getNumberOfArguments();
	unsigned k=0;
	log<<"------- GET GAUSSIAN SUPPORT --------\n"; 
	Matrix<double> mymatrix(ncv,ncv);
	for(unsigned i=0;i<ncv;i++){
		for(unsigned j=i;j<ncv;j++){
			mymatrix(i,j)=mymatrix(j,i)=hill.sigma[k]; // recompose the full inverse matrix
			k++;
		}
	}
        //
        // Reinvert so to have the ellipses 
        //
	Matrix<double> myinv(ncv,ncv);
	Invert(mymatrix,myinv);
	log<<"INVERSE \n"; 
        matrixOut(log,myinv);	
        // diagonalizes it
	Matrix<double> myautovec(ncv,ncv);
	vector<double> myautoval(ncv); //should I take this or their square root? 
	diagMat(myinv,myautoval,myautovec);
	for (unsigned i=0;i<ncv;i++){
		double cutoff=sqrt(2.0*DP2CUTOFF)*abs(sqrt(myautoval[0])*myautovec(i,0));
		//log<<"AUTOVAL "<<myautoval[0]<<" COMP "<<abs(myautoval[0]*myautovec(i,0)) <<" CUTOFF "<<cutoff<<"\n";
	  	nneigh.push_back( static_cast<unsigned>(ceil(cutoff/BiasGrid_->getDx()[i])) );
        }
 }else{
	 for(unsigned i=0;i<getNumberOfArguments();++i){
	  double cutoff=sqrt(2.0*DP2CUTOFF)*hill.sigma[i];
	  nneigh.push_back( static_cast<unsigned>(ceil(cutoff/BiasGrid_->getDx()[i])) );
 	}
 }
	log<<"------- END GET GAUSSIAN SUPPORT --------\n"; 
 return nneigh;
}

double BiasMetaD::getBiasAndDerivatives(const vector<double>& cv, double* der)
{
 double bias=0.0;
 if(!grid_){
  for(unsigned i=0;i<hills_.size();++i){
   bias+=evaluateGaussian(cv,hills_[i],der);
   //finite difference test 
   //finiteDifferenceGaussian(cv,hills_[i]);
  }
 }else{
  if(der){
   vector<double> vder(getNumberOfArguments());
   bias=BiasGrid_->getValueAndDerivatives(cv,vder);
   for(unsigned i=0;i<getNumberOfArguments();++i){der[i]=vder[i];}
  }else{
   bias=BiasGrid_->getValue(cv);
  }
 }
 return bias;
}

double BiasMetaD::evaluateGaussian
 (const vector<double>& cv, const Gaussian& hill, double* der)
{
 double dp2=0.0;
 double bias=0.0;
 if(hill.multivariate){ 
    unsigned k=0;
    unsigned ncv=cv.size(); 
    // recompose the full sigma from the upper diag cholesky 
    Matrix<double> mymatrix(ncv,ncv);
    for(unsigned i=0;i<ncv;i++){
		for(unsigned j=i;j<ncv;j++){
			mymatrix(i,j)=mymatrix(j,i)=hill.sigma[k]; // recompose the full inverse matrix
			k++;
		}
    }

    for(unsigned i=0;i<cv.size();++i){
	double dp_i=difference(i,hill.center[i],cv[i]);
        dp_[i]=dp_i;
    	for(unsigned j=i;j<cv.size();++j){
                  if(i==j){ 
              	  	 dp2+=dp_i*dp_i*mymatrix(i,j)*0.5; 
                  }else{ 
   		 	 double dp_j=difference(j,hill.center[j],cv[j]);
              	  	 dp2+=dp_i*dp_j*mymatrix(i,j) ;
                  }
        }
    } 
    if(dp2<DP2CUTOFF){
     bias=hill.height*exp(-dp2);
     if(der){
      for(unsigned i=0;i<cv.size();++i){
                double tmp=0.0;
                k=i;
                for(unsigned j=0;j<cv.size();++j){
                                tmp+=   dp_[j]*mymatrix(i,j)*bias;
                        }
                        der[i]-=tmp;
                }   
     }
    }
 }else{
    for(unsigned i=0;i<cv.size();++i){
     double dp=difference(i,hill.center[i],cv[i])*hill.invsigma[i];
     dp2+=dp*dp;
     dp_[i]=dp;
    }
    dp2*=0.5;
    if(dp2<DP2CUTOFF){
     bias=hill.height*exp(-dp2);
     if(der){
      for(unsigned i=0;i<cv.size();++i){der[i]+=-bias*dp_[i]*hill.invsigma[i];}
     }
    }
 }
 return bias;
}
=======
//void BiasMetaD::writeGaussian(const GaussianKernel& hill, FILE* file)
//{
// unsigned ncv=getNumberOfArguments();
// fprintf(hillsfile_, "%10.3f   ", getTimeStep()*getStep());
// for(unsigned i=0;i<ncv;++i){fprintf(file, "%14.9f   ", hill.center[i]);}
// if(hill.multivariate){  
//	 fprintf(file, " MV ");
//         // build the full matrix, invert it and do the cholesky decomp
//	 Matrix<double> mymatrix(ncv,ncv);
//         unsigned k=0;
//	 for(unsigned i=0;i<ncv;i++){
//		for(unsigned j=i;j<ncv;j++){
//			mymatrix(i,j)=mymatrix(j,i)=hill.sigma[k]; // recompose the full inverse matrix
//			k++;
//		}
//	 }
//         // invert it 
//         Matrix<double> invmatrix(ncv,ncv);
//         Invert(mymatrix,invmatrix);
//         // enforce symmetry
//	 for(unsigned i=0;i<ncv;i++){
//		for(unsigned j=i;j<ncv;j++){
//			invmatrix(i,j)=invmatrix(j,i);
//		}
//	 }
//        
//         // do cholesky so to have a "sigma like" number
//         Matrix<double> lower(ncv,ncv);	
//    	 cholesky(invmatrix,lower); // now this , in band form , is similar to the sigmas
//         // loop in band form 
//         k=0;
//         for (unsigned i=0;i<ncv;i++){
//              for (unsigned j=0;j<ncv-i;j++){
//	              fprintf(file, "%14.9f   ", lower(j+i,j));
//                      k++;
//              }
//         }
// }else{
//	 for(unsigned i=0;i<ncv;++i){fprintf(file, "%14.9f   ", hill.sigma[i]);}
// }
// double height=hill.height;
// if(welltemp_){height*=biasf_/(biasf_-1.0);}
// fprintf(file, "%14.9f   %4.3f \n",height,biasf_);
//}

//void BiasMetaD::addGaussian(const Gaussian& hill)
//{
// if(!grid_){hills_.push_back(hill);} 
// else{
//  unsigned ncv=getNumberOfArguments();
//  vector<unsigned> nneighb=getGaussianSupport(hill);
//  vector<unsigned> neighbors=BiasGrid_->getNeighbors(hill.center,nneighb);
//  vector<double> der(ncv);
//  vector<double> xx(ncv);
//  for(unsigned i=0;i<neighbors.size();++i){
//   unsigned ineigh=neighbors[i];
//   for(unsigned j=0;j<ncv;++j){der[j]=0.0;}
//   BiasGrid_->getPoint(ineigh,xx);   
//   double bias=evaluateGaussian(xx,hill,&der[0]);
//   BiasGrid_->addValueAndDerivatives(ineigh,bias,der);
//  }
// }
//}

//vector<unsigned> BiasMetaD::getGaussianSupport(const Gaussian& hill)
//{
// vector<unsigned> nneigh;
// // traditional or flexible hill? 
// if(hill.multivariate){
//	unsigned ncv=getNumberOfArguments();
//	unsigned k=0;
//	log<<"------- GET GAUSSIAN SUPPORT --------\n"; 
//	Matrix<double> mymatrix(ncv,ncv);
//	for(unsigned i=0;i<ncv;i++){
//		for(unsigned j=i;j<ncv;j++){
//			mymatrix(i,j)=mymatrix(j,i)=hill.sigma[k]; // recompose the full inverse matrix
//			k++;
//		}
//	}
//        //
//        // Reinvert so to have the ellipses 
//        //
//	Matrix<double> myinv(ncv,ncv);
//	Invert(mymatrix,myinv);
//	log<<"INVERSE \n"; 
//        matrixOut(log,myinv);	
//        // diagonalizes it
//	Matrix<double> myautovec(ncv,ncv);
//	vector<double> myautoval(ncv); //should I take this or their square root? 
//	diagMat(myinv,myautoval,myautovec);
//	for (unsigned i=0;i<ncv;i++){
//		double cutoff=sqrt(2.0*DP2CUTOFF)*abs(sqrt(myautoval[0])*myautovec(i,0));
//		//log<<"AUTOVAL "<<myautoval[0]<<" COMP "<<abs(myautoval[0]*myautovec(i,0)) <<" CUTOFF "<<cutoff<<"\n";
//	  	nneigh.push_back( static_cast<unsigned>(ceil(cutoff/BiasGrid_->getDx()[i])) );
//        }
// }else{
//	 for(unsigned i=0;i<getNumberOfArguments();++i){
//	  double cutoff=sqrt(2.0*DP2CUTOFF)*hill.sigma[i];
//	  nneigh.push_back( static_cast<unsigned>(ceil(cutoff/BiasGrid_->getDx()[i])) );
// 	}
// }
//	log<<"------- END GET GAUSSIAN SUPPORT --------\n"; 
// return nneigh;
//}

//double BiasMetaD::getBiasAndDerivatives(const vector<double>& cv, double* der)
//{
// double bias=0.0;
// if(!grid_){
//  for(unsigned i=0;i<hills_.size();++i){
//   bias+=hills_[i].evaluate( getArguments(), der, true );      //evaluateGaussian(cv,hills_[i],der);
//   //finite difference test 
//   //finiteDifferenceGaussian(cv,hills_[i]);
//  }
// }else{
//  if(der){
//   vector<double> vder(getNumberOfArguments());
//   bias=BiasGrid_->getValueAndDerivatives(cv,vder);
//   for(unsigned i=0;i<getNumberOfArguments();++i){der[i]=vder[i];}
//  }else{
//   bias=BiasGrid_->getValue(cv);
//  }
// }
// return bias;
//}

//double BiasMetaD::evaluateGaussian
// (const vector<double>& cv, const Gaussian& hill, double* der)
//{
// double dp2=0.0;
// double bias=0.0;
// if(hill.multivariate){ 
//    unsigned k=0;
//    unsigned ncv=cv.size(); 
//    // recompose the full sigma from the upper diag cholesky 
//    Matrix<double> mymatrix(ncv,ncv);
//    for(unsigned i=0;i<ncv;i++){
//		for(unsigned j=i;j<ncv;j++){
//			mymatrix(i,j)=mymatrix(j,i)=hill.sigma[k]; // recompose the full inverse matrix
//			k++;
//		}
//    }
//
//    for(unsigned i=0;i<cv.size();++i){
//	double dp_i=difference(i,hill.center[i],cv[i]);
//        dp_[i]=dp_i;
//    	for(unsigned j=i;j<cv.size();++j){
//                  if(i==j){ 
//              	  	 dp2+=dp_i*dp_i*mymatrix(i,j)*0.5; 
//                  }else{ 
//   		 	 double dp_j=difference(j,hill.center[j],cv[j]);
//              	  	 dp2+=dp_i*dp_j*mymatrix(i,j) ;
//                  }
//        }
//    } 
//    if(dp2<DP2CUTOFF){
//     bias=hill.height*exp(-dp2);
//     if(der){
//      for(unsigned i=0;i<cv.size();++i){
//                double tmp=0.0;
//                k=i;
//                for(unsigned j=0;j<cv.size();++j){
//                                tmp+=   dp_[j]*mymatrix(i,j)*bias;
//                        }
//                        der[i]-=tmp;
//                }   
//     }
//    }
// }else{
//    for(unsigned i=0;i<cv.size();++i){
//     double dp=difference(i,hill.center[i],cv[i])*hill.invsigma[i];
//     dp2+=dp*dp;
//     dp_[i]=dp;
//    }
//    dp2*=0.5;
//    if(dp2<DP2CUTOFF){
//     bias=hill.height*exp(-dp2);
//     if(der){
//      for(unsigned i=0;i<cv.size();++i){der[i]+=-bias*dp_[i]*hill.invsigma[i];}
//     }
//    }
// }
// return bias;
//}
*/
double BiasMetaD::getHeight(const vector<double>& cv)
{
 double height=height0_;
 if(welltemp_){
    double vbias=getPntrToComponent("bias")->get();      //=getBiasAndDerivatives(cv);
    height=height0_*exp(-vbias/(plumed.getAtoms().getKBoltzmann()*temp_*(biasf_-1.0)));
 } 
 return height;
}


void BiasMetaD::calculate()
{
//  unsigned ncv=getNumberOfArguments();
// vector<double> cv(ncv);
// for(unsigned i=0;i<ncv;++i){cv[i]=getArgument(i);}

  double ene=0;
  std::vector<double> der( getNumberOfArguments(),0.0 ); 

  if(!grid_){
     for(unsigned i=0;i<hills_.size();++i){
        ene+=hills_[i].evaluate( getArguments(), der, true );
     }
  } else {
     vector<double> cv(getNumberOfArguments());
     for(unsigned i=0;i<getNumberOfArguments();++i) cv[i]=getArgument(i);
     ene=BiasGrid_->getValueAndDerivatives( cv ,der );
  }
//  double ene=getBiasAndDerivatives(cv,der);
  getPntrToComponent("bias")->set(ene);

// set Forces 
  for(unsigned i=0;i<getNumberOfArguments();++i){
    double f=-der[i]; setOutputForce(i,f);
  }

//  delete [] der;
}

void BiasMetaD::update(){
  vector<double> cv(getNumberOfArguments());
  vector<double> thissigma;
  bool multivariate;

  // adding hills criteria (could be more complex though)
  bool nowAddAHill=(getStep()%stride_==0?true:false); 

  for(unsigned i=0;i<cv.size();++i){cv[i]=getArgument(i);}

  // if you use adaptive, call the FlexibleBin 
  if (adaptive_!=FlexibleBin::none){
       flexbin->update(nowAddAHill);
       multivariate=true;
  }else{
       multivariate=false;
  };

  if(nowAddAHill){ // probably this can be substituted with a signal
   // add a Gaussian
   double height=getHeight(cv);
   // use normal sigma or matrix form? 
   if (adaptive_!=FlexibleBin::none){	
	thissigma=flexbin->getInverseMatrix(); // returns upper diagonal inverse 
	//cerr<<"ADDING HILLS "<<endl;	
   }else{
	thissigma=sigma0_;    // returns normal sigma
   }
//   Gaussian newhill=Gaussian(cv,thissigma,height,multivariate);
//   addGaussian(newhill);
   if(!grid_){
      hills_.push_back( Kernel( cv,thissigma,kerneltype,height,false ) );
      hillsOfile_.printField("time", getTimeStep()*getStep());
      hills_[hills_.size()-1].print(argument_names, hillsOfile_);
      hillsOfile_.printField("biasf",biasf_ );
      hillsOfile_.printField();
   } else {
      Kernel kernel( cv, thissigma, kerneltype, height, false);
      BiasGrid_->addKernel( kernel );
      hillsOfile_.printField("time", getTimeStep()*getStep());
      kernel.print(argument_names, hillsOfile_);
      hillsOfile_.printField("biasf",biasf_ );
      hillsOfile_.printField();
   }
// print on HILLS file
//   writeGaussian(newhill,hillsOfile_);
  }
// dump grid on file
  if(wgridstride_>0&&getStep()%wgridstride_==0){
   BiasGrid_->writeToFile(gridfile_); 
  }
}

// void BiasMetaD::finiteDifferenceGaussian
//  (const vector<double>& cv, const Gaussian& hill)
// {
//  log<<"--------- finiteDifferenceGaussian: size "<<cv.size() <<"------------\n";
//  // for each cv
//  // first get the bias and the derivative
//  vector<double> oldder(cv.size()); 
//  vector<double> der(cv.size()); 
//  vector<double> mycv(cv.size()); 
//  mycv=cv; 
//  double step=1.e-6;
//  Random random; 
//  // just displace a tiny bit
//  for(unsigned i=0;i<cv.size();i++)log<<"CV "<<i<<" V "<<mycv[i]<<"\n";
//  for(unsigned i=0;i<cv.size();i++)mycv[i]+=1.e-2*2*(random.RandU01()-0.5);
//  for(unsigned i=0;i<cv.size();i++)log<<"NENEWWCV "<<i<<" V "<<mycv[i]<<"\n";
//  double oldbias=evaluateGaussian(mycv,hill,&oldder[0]);
//  for (unsigned i=0;i<mycv.size();i++){
//                double delta=step*2*(random.RandU01()-0.5);
//                mycv[i]+=delta;
//                double newbias=evaluateGaussian(mycv,hill,&der[0]);		
//                log<<"CV "<<i;
//                log<<" ANAL "<<oldder[i]<<" NUM "<<(newbias-oldbias)/delta<<" DIFF "<<(oldder[i]-(newbias-oldbias)/delta)<<"\n";
//                mycv[i]-=delta;
//  }
//  log<<"--------- END finiteDifferenceGaussian ------------\n";
// }

}
