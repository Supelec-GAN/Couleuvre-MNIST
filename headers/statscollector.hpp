#ifndef STATSCOLLECTOR_HPP
#define STATSCOLLECTOR_HPP

#include "headers/errorcollector.hpp"
#include "headers/CSVFile.h"

#include <vector>
#include <string>


namespace Stats
{

class StatsCollector
{
    public:
        StatsCollector(const std::string& CSVFileName = "resultat");

        ErrorCollector& operator[](unsigned int teachIndex);

        void exportData(bool mustProcessData = true);

        void writeCSV(std::string string, bool endRow = 0);
        void writeCSV(float number, bool endRow = 0);

    private:
        std::vector<ErrorCollector> mErrorStats;
        csvfile                     mCSV;
};

}

#endif // STATSCOLLECTOR_HPP
