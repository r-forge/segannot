
//////////////////////////////////
//                              //
//  FindOptimalSegmentations.h  //
//                              //
//////////////////////////////////

#ifndef FindOptimalSegmentations_h
#define FindOptimalSegmentations_h

void FindOptimalSegmentations(const double * x, const unsigned *sR, const unsigned *eR, const unsigned nMax,  
							  const unsigned pMax, unsigned * iDPath, double * cost);

#endif // FindOptimalSegmentations_h
