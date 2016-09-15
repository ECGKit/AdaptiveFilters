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

//using namespace std;
using namespace Eigen;

int main(int argc, char* argv[])
{
  StartTimer();

  double error, emse, msd;
  double MSE_theory = 0, EMSE_theory = 0;
  double mu = 5e-3;
  int realizations = 10;
  int realizationsPerThrd = 0;
  int iterations = 1000;
  double var_v = 0;
  int M = 2; //number of taps
  int nthreads = 0;
  int NUM_THREADS = 1;

  std::stringstream str_NUM_THREADS(argv[1]);
  str_NUM_THREADS >> NUM_THREADS;

  std::stringstream str_M(argv[2]);
  str_M >> M;

  std::stringstream str_realizations(argv[3]);
  str_realizations >> realizations;

  std::stringstream str_iterations(argv[4]);
  str_iterations >> iterations;

  std::stringstream str_mu(argv[5]);
  str_mu >> mu;

  std::stringstream str_var_v(argv[6]);
  str_var_v >> var_v;

  // Initializing variables
  double x{0};
  double d{0};
  double v{0};
  double wo{0.55};

  //Inititalizing Eigen matrices

  Eigen::Matrix<double, Dynamic, 1> wo_i; //Array - plant model initialized with M zeros
                                    wo_i.setZero(M, 1);

  for (int n=0; n < M; n++) //populating the arrays - regressor and noise
  {
    wo_i(n) = wo;
  }

  // Eigen::Matrix<double, Dynamic, 1> w_avg; //Array - weight vector averaged over realizations
  Eigen::Matrix<double, Dynamic, 1> w_avg_final; //Array - weight vector averaged over realizations
                                    w_avg_final.setZero(M, 1);
  Eigen::Matrix<double, Dynamic, Dynamic> W_avg; //Array - weight vector averaged over realizations
                                          W_avg.setZero(M, NUM_THREADS);
  Eigen::Matrix<double, Dynamic, Dynamic> MSE_avg; //MSE_avg
                                          MSE_avg.setZero(iterations, NUM_THREADS);
  Eigen::Matrix<double, Dynamic, Dynamic> EMSE_avg; //EMSE_avg
                                          EMSE_avg.setZero(iterations, NUM_THREADS);
  Eigen::Matrix<double, Dynamic, Dynamic> MSD_avg; //MSD_avg
                                          MSD_avg.setZero(iterations, NUM_THREADS);
  Eigen::Matrix<double, Dynamic, 1> MSE_avg_final; //MSE for ensemble average
                                    MSE_avg_final.setZero(iterations, 1);
  Eigen::Matrix<double, Dynamic, 1> EMSE_avg_final; //EMSE for ensemble average
                                    EMSE_avg_final.setZero(iterations, 1);
  Eigen::Matrix<double, Dynamic, 1> MSD_avg_final; //MSD for ensemble average
                                    MSD_avg_final.setZero(iterations, 1);

  omp_set_num_threads(NUM_THREADS); //Requesting NUM_THREADS threads to the host

  #pragma omp parallel
  {
    Eigen::Matrix<double, Dynamic, 1> u_i; //Regressor initialized with M zeros
                                      u_i.setZero(M, 1);
    // Eigen::Matrix<double, Dynamic, 1> wo_i; //Array - plant model initialized with M zeros
    //                                   wo_i.setZero(M, 1);
    Eigen::Matrix<double, Dynamic, 1> w_old; //Array - weight vector (old)
                                      w_old.setZero(M, 1);
    Eigen::Matrix<double, Dynamic, 1> w_new; //Array - weight vector (new)
                                      w_new.setZero(M, 1);
    Eigen::Matrix<double, Dynamic, 1> w_avg; //Array - weight vector averaged over realizations
                                      w_avg.setZero(M, 1);

    std::default_random_engine u;
    std::normal_distribution<double> normaldist(0,1); //Gaussian Distribution, mean=0, stddev=1;

    int j, id, nthrds;
    id = omp_get_thread_num();
    nthrds = omp_get_num_threads(); // checking how many threads were given by
                                    // the system.

    // std::cout << "Thread \n" << id << "\n" << std::endl;
    printf("Thread(%d)\n", id);

    realizationsPerThrd = realizations/nthrds;
    if (id == 0) nthreads = nthrds; // only executed by the master thread (id = 0)

    //#pragma omp for
    for (j = id; j < realizations; j = j + nthrds)
    {

        for (int n=0; n < M; n++) //populating the arrays - regressor and noise
        {
            x = normaldist(u); // generates normally distributed samples for noise
            u_i(n)  = x;
        }

        w_old.setZero(M, 1);
        w_new.setZero(M, 1);

        // Redefining MSE, EMSE, and MSD before each realization
        Eigen::Matrix<double, Dynamic, 1> MSE; //MSE
                                          MSE.setZero(iterations, 1);
        Eigen::Matrix<double, Dynamic, 1> EMSE; //EMSE
                                          EMSE.setZero(iterations, 1);
        Eigen::Matrix<double, Dynamic, 1> MSD; //MSD
                                          MSD.setZero(iterations, 1);

        for (int i = 0; i < iterations; ++i)
        {
          // Desirable output
          v = normaldist(u); // generates normally distributed samples for noise
          d = u_i.dot(wo_i) + sqrt(var_v)*v;

          //GA-LMS
          error = d - u_i.dot(w_old);
          emse = u_i.dot(wo_i - w_old);
          msd = (wo_i - w_old).norm();

  	      MSE(i) = pow(error,2); // Storing the squared error
          EMSE(i) = pow(emse,2); // Storing the squared excess error
        	MSD(i) = pow(msd,2); // Storing the squared mean deviation

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
        } //iterations

        // Ensemble average for each thread
        MSE_avg.col(id) += ((double) 1/realizationsPerThrd)*MSE;
        EMSE_avg.col(id) += ((double) 1/realizationsPerThrd)*EMSE;
        MSD_avg.col(id) += ((double) 1/realizationsPerThrd)*MSD;
        W_avg.col(id) += ((double) 1/realizationsPerThrd)*w_new;
    } //realizations

  } //pragma omp

  for (int nn = 0; nn < nthreads; nn++)
  {
    MSE_avg_final = MSE_avg_final + ((double) 1/nthreads)*MSE_avg.col(nn);
    EMSE_avg_final = EMSE_avg_final + ((double) 1/nthreads)*EMSE_avg.col(nn);
    MSD_avg_final = MSD_avg_final + ((double) 1/nthreads)*MSD_avg.col(nn);
    w_avg_final = w_avg_final + ((double) 1/nthreads)*W_avg.col(nn);
  }

  MSE_theory  = var_v + mu*M*var_v/(2-mu*M);
  EMSE_theory = mu*M*var_v/(2-mu*M);

  std::cout << "Number of taps = " << M << std::endl;
  std::cout << "Number of threads = " << NUM_THREADS << std::endl;
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
  std::ofstream output_MSE("./MSE_threads.out"); //file for saving MSE vector
  std::ofstream output_EMSE("./EMSE_threads.out"); //file for saving EMSE vector
  std::ofstream output_MSD("./MSD_threads.out"); //file for saving MSD vector
  std::ofstream output_MSE_final("./MSE.out"); //file for saving MSE vector
  std::ofstream output_EMSE_final("./EMSE.out"); //file for saving EMSE vector
  std::ofstream output_MSD_final("./MSD.out"); //file for saving MSD vector

  // Saving MSE, EMSE, and MSD
  for(int k = 0; k < iterations; k++){
    output_MSE << MSE_avg.row(k) << "\n";
    output_EMSE << EMSE_avg.row(k) << "\n";
    output_MSD << MSD_avg.row(k) << "\n";
    output_MSE_final << MSE_avg_final.row(k) << "\n";
    output_EMSE_final << EMSE_avg_final.row(k) << "\n";
    output_MSD_final << MSD_avg_final.row(k) << "\n";
  }

  std::ofstream output_EMSE_theory("./EMSE_theory.out"); //saving EMSE_theory bound
  output_EMSE_theory << EMSE_theory << '\n';

  std::ofstream output_MSE_theory("./MSE_theory.out"); //saving MSE_theory bound
  output_MSE_theory << MSE_theory << '\n';

  // std::ofstream output_w("./w_avg.out"); //saving w_avg array
  // for (int n = 0; n < M; ++n)
  // {
  //    output_w << w_avg[n] << '\n'; // saves w_avg. Each line is an entry
  // }

  printf("Elapsed Time (ms) = %g\n", GetTimer());
}
