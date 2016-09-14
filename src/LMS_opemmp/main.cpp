#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <sstream>
#include <iterator> // for ostream_iterator
#include <math.h>       /* log10 */
#include <stdio.h>
#include <omp.h> // openmp
#include "Eigen"
#include "gnuplot-iostream.h"

#include "timer.h"

#define NUM_THREADS 2

//using namespace std;
using namespace Eigen;

int main(int argc, char* argv[])
{
  StartTimer();

  double error, emse, msd;
  double MSE_theory = 0, EMSE_theory = 0;
  double mu = 5e-3;
  int realizations = 10;
  int realizationsPerThrd = 10;
  int iterations = 1000;
  double var_v = 0;
  int M = 2; //number of taps
  int nthreads = 1;

  std::stringstream str_M(argv[1]);
  str_M >> M;

  std::stringstream str_realizations(argv[2]);
  str_realizations >> realizations;

  std::stringstream str_iterations(argv[3]);
  str_iterations >> iterations;

  std::stringstream str_mu(argv[4]);
  str_mu >> mu;

  std::stringstream str_var_v(argv[5]);
  str_var_v >> var_v;


  // Initializing variables
  double x{0};
  double d{0};
  double v{0};
  double wo{0.55};

  //Inititalizing Eigen matrices
  Eigen::Matrix<double, Dynamic, 1> u_i; //Array of regressors initialized with M zeros
                                    u_i.setZero(M, 1);
  Eigen::Matrix<double, Dynamic, 1> wo_i; //Array - plant model initialized with M zeros
                                    wo_i.setZero(M, 1);
  Eigen::Matrix<double, Dynamic, 1> w_old; //Array - weight vector (old)
  Eigen::Matrix<double, Dynamic, 1> w_new; //Array - weight vector (new)
  Eigen::Matrix<double, Dynamic, 1> w_avg; //Array - weight vector averaged over realizations
  Eigen::Matrix<double, Dynamic, 1> w_avg_final; //Array - weight vector averaged over realizations
  Eigen::Matrix<double, Dynamic, Dynamic> W_avg; //Array - weight vector averaged over realizations
  Eigen::Matrix<double, Dynamic, 1> resource; //Array - resource

  std::default_random_engine u;
  std::normal_distribution<double> normaldist(0,1); //Gaussian Distribution, mean=0, stddev=1;

  std::vector<double> MSE_avg;
  std::vector<double> EMSE_avg;
  std::vector<double> MSD_avg;

  // Resizing the following vectors to allocate memory. Otherwise
  // a 'segmentation fault'
  MSE_avg.resize(iterations);
  EMSE_avg.resize(iterations);
  MSD_avg.resize(iterations);

  W_avg.setZero(M, NUM_THREADS);
  w_avg_final.setZero(M, 1);
  omp_set_num_threads(NUM_THREADS);

  #pragma omp parallel
  {

    int j, id, nthrds;
    id = omp_get_thread_num();
    nthrds = omp_get_num_threads(); // checking how many threads were given by
                                    // the system.

    // std::cout << "Thread \n" << id << "\n" << std::endl;
    printf("Thread(%d)\n", id);

    realizationsPerThrd = realizations/nthrds;
    if (id == 0) nthreads = nthrds; // only executed by the master thread (id = 0)

    for (j = id; j < realizations; j = j + nthrds)
    {

        // std::cout << "Realization " << j+1
        //           << " of " << realizations
        //           << " at thread "<< id << std::endl;

        for (int n=0; n < M; n++) //populating the arrays - regressor and noise
        {
            x = normaldist(u); // generates normally distributed samples for noise
            u_i(n)  = x;
            wo_i(n) = wo;
        }

        w_old.setZero(M, 1);
        w_new.setZero(M, 1);
        w_avg.setZero(M, 1);

        //std::cout << "wo_i = \n" << wo_i << std::endl;

        // Redefining MSE_galms and EMSE_galms before each realization
        std::vector<double> MSE;
        std::vector<double> EMSE;
        std::vector<double> MSD;

        for (int i = 0; i < iterations; ++i)
        {
          // Desirable output
          v = normaldist(u); // generates normally distributed samples for noise
          d = u_i.dot(wo_i) + sqrt(var_v)*v;

          //y = array_prod(reverse_array(u_i),R[0]);

          //GA-LMS
          error = d - u_i.dot(w_old);
          emse = u_i.dot(wo_i - w_old);
          msd = (wo_i - w_old).norm();

  	      MSE.push_back(pow(error,2)); // .push_back shifts the previous content of the vector
          EMSE.push_back(pow(emse,2)); // .push_back shifts the previous content of the vector
        	MSD.push_back(pow(msd,2)); // .push_back shifts the previous content of the vector

          w_new = w_old + mu*u_i*error;
          w_old = w_new;

          /*
          //Regenerating regressor - shift register =================
          for (size_t j = 0; j < 8; ++j)
          {
              x[j] = normaldist(u);
              v[j] = normaldist(u);
          }

          std::rotate(u_i.begin(), u_i.begin()+1, u_i.end());
          //std::cout << "regressor_after_rotation = " << std::endl;
          //for (int n = 0; n < M; ++n)
          //std::cerr << u_i[n] << std::endl;

          u_i.at(M-1) = x;//replaces element in the back of u_i
          //std::cout << "regressor_after_replacement = " << std::endl;
          //for (int n = 0; n < M; ++n)
          //std::cerr << u_i[n] << std::endl;
          //=========================================================
          */

          //Regenerating regressor - no shift register ==============
          for (int n=0; n < M; n++)
          {
              x = normaldist(u);
              u_i(n)  = x;
          }

          //std::cout << "new regressor = " << std::endl;
          //for (int n = 0; n < M; ++n)
          //std::cerr << u_i[n] << std::endl;
          //=========================================================
        }

        for (int r = 0; r < iterations; ++r)
        {
          MSE_avg[r] = MSE_avg[r] + ((double) 1/realizationsPerThrd)*MSE[r];
          EMSE_avg[r] = EMSE_avg[r] + ((double) 1/realizationsPerThrd)*EMSE[r];
          MSD_avg[r] = MSD_avg[r] + ((double) 1/realizationsPerThrd)*MSD[r];
        }

        //w_new.setOnes(M);

        //w_avg = w_avg + ((double) 1/realizationsPerThrd)*w_new; // averaging estimated w
        W_avg.col(id) += ((double) 1/realizationsPerThrd)*w_new;
        std::cout << "W_avg.col(id) for each thread = \n" << W_avg.col(id) << std::endl;
    } //realizations
    //W_avg.col(id) = w_avg;

  } //pragma omp

  for (int nn = 0; nn < NUM_THREADS; nn++)
  {
    w_avg_final = w_avg_final + ((double) 1/NUM_THREADS)*W_avg.col(nn);
  }


  MSE_theory  = var_v + mu*M*var_v/(2-mu*M);
  EMSE_theory = mu*M*var_v/(2-mu*M);

  std::cout << "Number of taps = " << M << std::endl;
  std::cout << "Realizations = " << realizations << std::endl;
  std::cout << "Realizations per Thread = " << realizationsPerThrd << std::endl;
  std::cout << "Iterations = " << iterations << std::endl;
  std::cout << "Step size = " << mu << std::endl;
  std::cout << "Measurement noise variance= " << var_v << std::endl;
  std::cout << "W_avg = \n" << W_avg << std::endl;
  std::cout << "Optimal wo = \n" << wo_i << std::endl;
  //std::cout << "Estimated w = \n" << w_new << std::endl;
  std::cout << "w averaged over realizations = \n" << w_avg_final << std::endl;

  // SAVING ==============================================================================
  std::ofstream output_MSE("./MSE.out"); //saving MSE vector
  std::ostream_iterator<double> output_iterator_MSE(output_MSE, "\n");
  std::copy(MSE_avg.begin(), MSE_avg.end(), output_iterator_MSE);

  std::ofstream output_EMSE("./EMSE.out"); //saving EMSE vector
  std::ostream_iterator<double> output_iterator_EMSE(output_EMSE, "\n");
  std::copy(EMSE_avg.begin(), EMSE_avg.end(), output_iterator_EMSE);

  std::ofstream output_MSD("./MSD.out"); //saving MSD vector
  std::ostream_iterator<double> output_iterator_MSD(output_MSD, "\n");
  std::copy(MSD_avg.begin(), MSD_avg.end(), output_iterator_MSD);

  std::ofstream output_EMSE_theory("./EMSE_theory.out"); //saving EMSE_theory bound
  output_EMSE_theory << EMSE_theory << '\n';

  std::ofstream output_MSE_theory("./MSE_theory.out"); //saving MSE_theory bound
  output_MSE_theory << MSE_theory << '\n';

  std::ofstream output_w("./w_avg.out"); //saving w_avg array
  for (int n = 0; n < M; ++n)
  {
     output_w << w_avg[n] << '\n'; // saves w_avg. Each line is an entry
  }

  printf("Elapsed Time (ms) = %g\n", GetTimer());
}
