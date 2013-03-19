
//////////////////////////////////
//                              //
//  FindOptimalSegmentations.h  //
//                              //
//////////////////////////////////

#ifndef FindOptimalSegmentations_h
#define FindOptimalSegmentations_h

void FindOptimalSegmentations(
    const double * x, const unsigned *sR, const unsigned *eR, 
    const unsigned nMax, const unsigned pMax, 
    unsigned * iDPath, double * cost);

int bases(
    const double * x, const unsigned * base,
    const unsigned *first_base, const unsigned *last_base, 
    unsigned *eR, unsigned *sR, 
    const unsigned nMax, const unsigned n_regions, 
    unsigned * iDPath, double *cost);

#define ERROR_BASES_NOT_INCREASING 1
#define ERROR_REGIONS_NOT_INCREASING 2
#define ERROR_LAST_BEFORE_FIRST 3

#endif // FindOptimalSegmentations_h
