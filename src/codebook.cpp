#include "utils.hpp"
#include "nmtf.hpp"
#include <boost/program_options.hpp>
#include <iostream>

struct Arguments {
    std::string source, target;
    double nmtfConv, nmtfRate;
    int nClustersU, nClustersI, rD1, rD2;
};


Arguments getArgs(int argc, char**argv) {
    Arguments args;
    try {
        namespace po = boost::program_options;
        po::options_description required("Required arguments:");
        required.add_options()
            ("source", po::value<std::string>(&args.source) -> required(), "source.txt")
            ("target", po::value<std::string>(&args.target) -> required(), "test.txt");

        po::options_description optional("Optional arguments:");
        optional.add_options()
            ("help", "Print help message.")
            ("nmtfRate", po::value<double>(&args.nmtfRate) -> default_value(0.0001), "learning rate for nmtf")
            ("converge", po::value<double>(&args.nmtfConv) -> default_value(10), "converge criteria for nmtf")
            ("rD1", po::value<int>(&args.rD1) -> default_value(50000), "dimention of rating matrix")             
            ("rD2", po::value<int>(&args.rD2) -> default_value(5000), "dimention of rating matrix")             
            ("nClustersU", po::value<int>(&args.nClustersU) -> default_value(10), "number of user clusters")
            ("nClustersI", po::value<int>(&args.nClustersI) -> default_value(10), "number of item clusters");

        po::options_description desc("===== Codebook =====");
        desc.add(required).add(optional);
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);

        if (vm.count("help")) {
            std::cout << desc << std::endl;
            exit(0);
        }

        po::notify(vm);
    } catch (boost::program_options::error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        exit(-1);
    }

    return args;
}

int main(int argc, char**argv){
    Arguments args = getArgs(argc, argv);

    Eigen::MatrixXd sourceRate(args.rD1, args.rD2);
    Eigen::MatrixXd sourceMask(args.rD1, args.rD2);
    loadMatrix(args.source, sourceRate, sourceMask);

    nmtf(args.nmtfRate, args.nmtfConv, sourceRate, sourceMask);
    
    return 0;
}
