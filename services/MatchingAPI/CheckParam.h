#ifndef MATCHINGAPI_CHECKPARAM_H
#define MATCHINGAPI_CHECKPARAM_H

#include "Includes.h"

template <class T> bool ifin(double param, T * range, int range_size)
{
    bool flag = false;
    for (int i = 0; i < range_size; i++)
    {
        if (param == range[i])
        {
            flag = true;
            break;
        }
    }
    return flag;
}

template <class T> void CheckParam(map<string, double> params, char* paramName, T * param, T * range, int range_size)
{
    try
    {
        double buf = params.at(paramName);
        bool flag = ifin(buf, range, range_size);
        if (!flag)
        {
            cout << endl << "Parameter " << paramName << " is out of range. Check readme for details. Default value will be used.";
        }
        else {*param = (T)buf;}
    }
    catch (std::out_of_range) {cout << endl << "Default value for " << paramName << " will be used";}
    cout << endl << "Parameter " << paramName << " is set to " << *param;
}

template <class T> void CheckParam_no_ifin(map<string, double> params, char* paramName, T * param, T min, T max)
{
    try
    {
        double buf = params.at(paramName);
        if (buf < min || buf > max)
        {
            cout << endl << "Parameter " << paramName << " is out of range. Check readme for details. Default value will be used.";
        }
        else {*param = (T)buf;}
    }
    catch (std::out_of_range) {cout << endl <<"Default value for " << paramName << " will be used";}
    cout << endl << "Parameter " << paramName << " is set to " << *param;
}

#endif //MATCHINGAPI_CHECKPARAM_H
