#include <iostream>
#include <fstream>
#include "gpf.h"

using namespace std;
using namespace PF;

//#define FILEIO

const double proc_noise_sd  = sqrt(10.0); 
const double obsv_noise_sd = 1.0; 
const uint T = 100;   
const uint Nsamples = 1000;
const uint Nx = 1;
const uint Nz = 1;
uint k = 0;
//----------------------------
// Process Equation
// xn : process noise
void process(std::vector<double> &xk, const std::vector<double> &xkm1, void* data)
{
  gsl_rng *r = (gsl_rng*)data;

  double un = gsl_ran_gaussian(r, proc_noise_sd);
  xk[0] = 0.5*xkm1[0] + 25*xkm1[0]/(1+xkm1[0]*xkm1[0]) + 8*cos(1.2*k) + un ;
}
//-------------------------
// Observation Equation
// vn: measurement noise 
void observation(std::vector<double> &zk, const std::vector<double> &xk, void* data)
{
  gsl_rng *r = (gsl_rng*)data;
  double vn = gsl_ran_gaussian(r, obsv_noise_sd);
  zk[0] = xk[0]*xk[0]/20.0 + vn;
}
//-------------------------------------------
// Likelihood: P(zk | xk)
// ----------------------------------------
double likelihood(const std::vector<double> &z, const std::vector<double> &zhat, void* data)
{
  double prod = 1.0;
  for(uint i = 0; i < Nz; ++i)
  {
    double e = z[i] - zhat[i];
    prod = prod * gsl_ran_gaussian_pdf(e, obsv_noise_sd);
  }

  return prod;
}
//-----------------------------------------------------
int main()
{
  gsl_rng *rg;
  long seed = time(NULL)*getpid();
  rg = gsl_rng_alloc(gsl_rng_rand48);
  gsl_rng_set(rg,seed);

  //Re-sample criterion
  float resample_percentage = 0.5;
  uint Nt = ceil(resample_percentage * Nsamples);

  // Initial variance in the state estimate
  double V = 2.0;


  std::vector<double> x(1);
  std::vector<double> z(1);
  std::vector<double> xf(1);

  // Initial value 
  x[0] = 0.1;


  // Create a pointCloud 
  PF::pf  pointCloud(Nsamples, Nx, Nz, SYSTEMATIC);

  ofstream f1("data.txt");
  for(k = 0; k < T; ++k)
  {
    if(k == 0)
    {
      pointCloud.initialize(k, x[0], sqrt(V));
      observation(z, x, (void*)rg);

#ifdef FILEIO
      std::vector<double> xp(Nx);
      std::vector<double> zp(Nz);
      std::vector<double> w(Nsamples);
      cout << "Initial Distribution of states" << endl;
      ofstream f1("init_x.txt");
      for(uint i = 0; i < Nsamples; ++i)
      {
        w[i] = pointCloud.getParticleState(xp, zp, i);
        f1 << xp[0] << endl;
      }
      f1.close();

      cout << "Use following command to see plot:" << endl;
      cout << "octave plot1.m" << endl;
      getchar();
#endif
    }
    else // k > 0
    {
      //Actual values
      process(x, x, (void*)rg);             // p(xk | xkm1)
      observation(z, x, (void*)rg);         // p(yk | xk)

      // Note that for us, only measurement is available
      // Estimate the states using particle Filter
      pointCloud.particleFilterUpdate(process, observation,likelihood,z, 0); // Don't resample  here


#ifdef FILEIO
      std::vector<double> xp(Nx);
      std::vector<double> zp(Nz);
      std::vector<double> w(Nsamples);

      cout << "Distribution before sampling" << endl;
      ofstream f2("before_sampling.txt");
      f2 << x[0] << "\t" << z[0] << "\t" << k << endl << endl;

      for(uint i = 0; i < Nsamples; ++i)
      {
        w[i] = pointCloud.getParticleState(xp, zp, i);
        f2 << xp[0] << "\t" << zp[0] << "\t" << w[i] << endl;
      }
      f2.close();

      cout << "Use following command to see plot:" << endl;
      cout << "octave plot2.m" << endl;
      getchar();
#endif

      float neff = pointCloud.getEffectivePopulation();
      cout << "k = " << pointCloud.getItnNum() << "\t Neff = " << neff << "\t Nt = " << Nt << "\t";

      pointCloud.filterOutput(xf);

      if(ceil(neff) < Nt)
      {
        cout << "Resampling ..." ;
        pointCloud.resample();

#ifdef FILEIO
        std::vector<double> xp(Nx);
        std::vector<double> zp(Nz);
        std::vector<double> w(Nsamples);

        ofstream f3("after_sampling.txt");
        f3 << xf[0] << "\t" << k << endl ;
        for(uint i = 0; i < Nsamples; ++i)
        {
          w[i] = pointCloud.getParticleState(xp,zp, i);
          f3 << xp[0] << "\t" << w[i] << endl;
        }
        cout << "Use following command to see plot:" << endl;
        cout << "octave plot3.m" << endl;
        f3.close();
        getchar();
#endif
      }
      cout << endl;
        
      f1 << k << "\t" << x[0] << "\t" << z[0] << "\t" << xf[0] << endl;
    }
  }
  f1.close();

  gsl_rng_free(rg);

  return 0;
}
