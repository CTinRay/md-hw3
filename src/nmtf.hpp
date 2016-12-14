#include <eigen3/Eigen/Dense>

#ifndef NMTF_HPP
#define NMTF_HPP

void nmtf(double rate, double conv,
          const Eigen::MatrixXd x,
          const Eigen::MatrixXd mask,
          Eigen::MatrixXd&f,
          Eigen::MatrixXd&s,
          Eigen::MatrixXd&g);
#endif
