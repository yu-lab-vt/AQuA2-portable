// Stable C ABI around unchanged GPL upstream kernels. Python serializes calls
// because each original kernel uses process-global scratch state.
#include "mex.h"
#include <Python.h>
#include <algorithm>
#include <mutex>
#ifdef AQUA_BILCO_CORE
#include "vendor/BILCOMex.cpp"
#else
#include "vendor/DTW_Edge_input.cpp"
#endif
#ifdef _WIN32
#define AQUA_EXPORT extern "C" __declspec(dllexport)
#else
#define AQUA_EXPORT extern "C" __attribute__((visibility("default")))
#endif
static std::mutex call_mutex;
static thread_local std::string last_error;
#ifdef AQUA_BILCO_CORE
static PyModuleDef module={PyModuleDef_HEAD_INIT,"_bilco_core",nullptr,-1,nullptr};
PyMODINIT_FUNC PyInit__bilco_core() { return PyModule_Create(&module); }
#else
static PyModuleDef module={PyModuleDef_HEAD_INIT,"_dtw_core",nullptr,-1,nullptr};
PyMODINIT_FUNC PyInit__dtw_core() { return PyModule_Create(&module); }
#endif
AQUA_EXPORT const char* aqua_error() { return last_error.c_str(); }
AQUA_EXPORT int aqua_run(int n,int t1,int t2,int pair_count,
    const double* ref_data,const double* tst_data,const double* pair_data,
    const double* cut_data,int window,int absolute,double* output,double* flow_output) {
    std::lock_guard<std::mutex> lock(call_mutex);
    mxArray* outputs[2]={nullptr,nullptr};
    last_error.clear();
    try {
#ifdef AQUA_BILCO_CORE
        mxArray ref(n,t1,ref_data),tst(n,t2,tst_data),pairs(pair_count,3,pair_data);
        mxArray cut(n,t1-1,cut_data),win(1,1),metric(0,0);
        win.values[0]=window;metric.text=absolute?"absolute":"squared";
        // The source reserves 16 neighbors. Only increase storage capacity for
        // denser graphs; no graph edges, ordering, capacities or math change.
        std::vector<int> degrees(n,0);
        for(int p=0;p<pair_count;++p) {
            ++degrees[(int)pair_data[p]-1];++degrees[(int)pair_data[p+pair_count]-1];
        }
        maxNeiNum=16;
        for(int degree:degrees) if(degree>maxNeiNum) maxNeiNum=degree;
        const mxArray* inputs[]={&ref,&tst,&pairs,&cut,&win,&metric};
        mexFunction(2,outputs,6,inputs);
        *flow_output=outputs[1]->values[0];
#else
        mxArray distance(t2,t1,ref_data);
        const mxArray* inputs[]={&distance};
        mexFunction(1,outputs,1,inputs);
#endif
        std::copy(outputs[0]->values.begin(),outputs[0]->values.end(),output);
        delete outputs[0];delete outputs[1];return 0;
    } catch(const std::exception& error) { last_error=error.what(); }
      catch(...) { last_error="Unknown native failure"; }
    delete outputs[0];delete outputs[1];return 1;
}
