#include "utils.hpp"
#include <fstream>
#include <cassert>

void loadMatrix(const std::string filename,
                Eigen::MatrixXd&matrix, Eigen::MatrixXd&mask){
    assert(matrix.rows() == mask.rows());
    assert(matrix.cols() == mask.cols());
    
    std::fstream f;
    f.open(filename, std::ios::in);
    int x, y;
    double rate;
    f >> x >> y >> rate;
    while (!f.eof()) {
        assert(x < matrix.rows());
        assert(y < matrix.cols());
        matrix(x, y) = rate;
        mask(x, y) = 1;
        f >> x >> y >> rate;   
    }
}
