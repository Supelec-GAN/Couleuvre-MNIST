#include "headers/statscollector.hpp"

#include <stdexcept>

Stats::StatsCollector::StatsCollector(const std::string& CSVFileName)
: mCSV(CSVFileName + ".csv")
{}

Stats::ErrorCollector& Stats::StatsCollector::operator[](unsigned int teachIndex)
{
    if(teachIndex > mErrorStats.size())
        throw std::logic_error("StatsCollector::operator[] - Erreur : Indice d'apprentissage trop grand");

    if(teachIndex == mErrorStats.size())
        mErrorStats.push_back(ErrorCollector());

    return  mErrorStats[teachIndex];
}

void Stats::StatsCollector::exportData(bool mustProcessData)
{
    if(!mustProcessData)
        throw std::logic_error("Not implemented yet");

    mCSV << "Teach index" << "Mean" << "Deviation" << "Confidence Range" << endrow;
    for (unsigned int index{0}; index < mErrorStats.size(); ++index)
    {
        ErrorCollector::StatisticData data{mErrorStats[index].processData()};

        mCSV << index << data.mean << data.deviation << data.confidenceRange << endrow;
    }
}
