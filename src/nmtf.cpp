#include "nmtf.hpp"
#include <eigen3/Eigen/Dense>

void nmtf(double rate, double conv,
                     int d1, int d2,
                     const Eigen::MatrixXd x,
                     const Eigen::MatrixXd mask,
                     Eigen::MatrixXd&f,
                     Eigen::MatrixXd&s,
                     Eigen::MatrixXd&g){
    double d = std::numeric_limits<double>::infinity();
    f.setRandom(x.rows(), d1);
    g.setRandom(x.cols(), d1);
    s.setRandom(x.rows(), x.cols());
    while (d > conv * conv) {
        d = 0;
        for (auto i = 0u; i < x.rows(); ++i) {
            for (auto j = 0u; j < x.cols(); ++j) {
                if (mask(i, j) != 0) {
                    auto e = x(i, j) - f.row(i) * s * g.row(j).transpose();
                    auto dfi = - e * s * g.row(j).transpose();
                    auto dgi = - e * f.row(i) * s;
                    auto ds  = - e * f.row(i) * g.row(j);
                    d += dfi.norm() + dgi.norm() + ds.norm();
                    f.row(i).array() -= dfi.array() * rate;
                    g.row(i).array() -= dgi.array() * rate;
                    s -= ds * rate;
                }
            }
        }
    }
}
