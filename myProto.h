#ifndef __MY__PROTO_H__
#define __MY__PROTO_H__

enum numbers{ROOT = 0, FIRST_GROUP_SIZE = 9, SECOND_GROUP_SIZE = 11, BUF_SIZE = 3000, BUFFER = 500000};

int computeOnGPU(int* weight, char* firstSeq, char** sequences, int numOfSeqs, int** maxScore, int** maxOffset, int** maxMutant, int* sizeMat);

#endif