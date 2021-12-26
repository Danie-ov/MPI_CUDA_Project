#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "myProto.h"
#include <stdio.h>
#include <string.h>



__device__ int checkStrChr(char* str, char ch)
{
    while(*str != NULL)
    {
        if(*str == ch)
            return 1;
        
        str++;
    }
    return 0;
}

__device__ int checkFirstGroupGPU(char a, char b)
{
   
   char* firstGroup[FIRST_GROUP_SIZE] = { "NDEQ", "NEQK", "STA", "MILV", "QHRK", "NHQK", "FYW", "HY", "MILF" };
   int i;
   for(i=0; i < FIRST_GROUP_SIZE; i++)
   {
      if(checkStrChr(firstGroup[i], a) && checkStrChr(firstGroup[i], b))
            return 1;
   }
   return 0;
}

__device__ int checkSecondGroupGPU(char a, char b)
{

    char* secondGroup[SECOND_GROUP_SIZE] = { "SAG", "ATV", "CSA", "SGND", "STPA", "STNK", "NEQHRK", "NDEQHK", "SNDEQK", "HFY", "FVLIM" };
   int i;
   for(i=0; i < SECOND_GROUP_SIZE; i++)
   {
      if(checkStrChr(secondGroup[i], a) && checkStrChr(secondGroup[i], b))
            return 1;
   }
   return 0;
}


__global__  void calcScore(int* weight, char* firstSeq, char* sequence, int firstSeqLen, int secondSeqLen, int* scores, int* offsets, int* mutants) {
    
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    int i, score = 0, offset = id / secondSeqLen, mutant = id % secondSeqLen + 1;
    char a, b;
    
    for(i=0; i <= secondSeqLen; i++)
    {
        a = firstSeq[i+offset];
        if(i < mutant)
            b = sequence[i];
            
        else if(i > mutant)
            b = sequence[i-1];
            
        if(i != mutant)
        {
            if(a == b)
                score += weight[0];
            
            else if(checkFirstGroupGPU(a, b))
            {
                score -= weight[1];

            }
            
            else if(checkSecondGroupGPU(a, b))
                score -= weight[2];
            else
                score -= weight[3];
        }    
    }
    scores[id] = score;
    offsets[id] = offset;
    mutants[id] = mutant;
}

int computeOnGPU(int* weight, char* firstSeq, char** sequences, int numOfSeqs, int** maxScore, int** maxOffset, int** maxMutant, int* sizeMat) {
    
    int* dev_weight = 0, *dev_max_score = 0, *dev_max_offset = 0, *dev_max_mutant = 0;
    char* dev_firstSeq = 0;
    int firstSeqLen = strlen(firstSeq), i;
    char* dev_sequence = 0;
    int seqLen = strlen(sequences[0]);
    cudaError_t cuda_status;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int numThreadsPerBlock, numBlocks, extraBlock;

    cuda_status = cudaMalloc((void**)&dev_weight, sizeof(int)*4);
    if(cuda_status != cudaSuccess)
    {
        printf("CUDA STATUS1 ERROR!\n");
        cudaFree(dev_max_score);
        cudaFree(dev_max_offset);
        cudaFree(dev_max_mutant);
        cudaFree(dev_weight);
        return EXIT_FAILURE;
    }

    cuda_status = cudaMemcpy(dev_weight, weight, 4*sizeof(int), cudaMemcpyHostToDevice);
    if(cuda_status != cudaSuccess)
    {
        printf("CUDA STATUS2 ERROR!\n");
        cudaFree(dev_max_score);
        cudaFree(dev_max_offset);
        cudaFree(dev_max_mutant);
        cudaFree(dev_weight);
        return EXIT_FAILURE;
    }

    cuda_status = cudaMalloc((void**)&dev_firstSeq, sizeof(char)*firstSeqLen);
    if(cuda_status != cudaSuccess)
    {
        printf("CUDA STATUS3 ERROR\n");
        cudaFree(dev_max_score);
        cudaFree(dev_max_offset);
        cudaFree(dev_max_mutant);
        cudaFree(dev_weight);
        return EXIT_FAILURE;
    }

    cuda_status = cudaMemcpy(dev_firstSeq, firstSeq, firstSeqLen, cudaMemcpyHostToDevice);
    if(cuda_status != cudaSuccess)
    {
        printf("CUDA STATUS4 ERROR!\n");
        cudaFree(dev_max_score);
        cudaFree(dev_max_offset);
        cudaFree(dev_max_mutant);
        cudaFree(dev_weight);
        cudaFree(dev_firstSeq);
        return EXIT_FAILURE;
    }

    for(i=0; i < numOfSeqs; i++)
    {
        int secondSeqLen = strlen(sequences[i]);
        int size = (firstSeqLen - secondSeqLen) * secondSeqLen;

        numThreadsPerBlock = prop.maxThreadsPerBlock < size ? prop.maxThreadsPerBlock : size;
        numBlocks = size / numThreadsPerBlock;
        extraBlock = size % numThreadsPerBlock != 0;

        cuda_status = cudaMalloc((void**)&dev_max_score, sizeof(int)*size);
        if(cuda_status != cudaSuccess)
        {
            printf("CUDA SCORE ERROR!\n");
            return EXIT_FAILURE;
        }
        cuda_status = cudaMalloc((void**)&dev_max_offset, sizeof(int)*size);
        if(cuda_status != cudaSuccess)
        {
            printf("CUDA OFFSET ERROR!\n");
            cudaFree(dev_max_score);
            return EXIT_FAILURE;
        }
        cuda_status = cudaMalloc((void**)&dev_max_mutant, sizeof(int)*size);
        if(cuda_status != cudaSuccess)
        {
            printf("CUDA MUTANT ERROR!\n");
            cudaFree(dev_max_score);
            cudaFree(dev_max_offset);
            return EXIT_FAILURE;
        }

        cuda_status = cudaMalloc((void**)&dev_sequence, sizeof(char)*secondSeqLen);
        if(cuda_status != cudaSuccess)
        {
            printf("CUDA STATUS6 ERROR!\n");
            cudaFree(dev_weight);
            cudaFree(dev_firstSeq);
            return EXIT_FAILURE;
        }

        cuda_status = cudaMemcpy(dev_sequence, sequences[i], sizeof(char)*secondSeqLen, cudaMemcpyHostToDevice);
        if(cuda_status != cudaSuccess)
        {
            printf("CUDA STATUS7 ERROR!\n");
            cudaFree(dev_weight);
            cudaFree(dev_firstSeq);
            return EXIT_FAILURE;
        }

        calcScore<<<numBlocks+extraBlock, numThreadsPerBlock>>>(dev_weight, dev_firstSeq, dev_sequence, firstSeqLen, seqLen, dev_max_score, dev_max_offset, dev_max_mutant);

        cuda_status = cudaDeviceSynchronize();
        if(cuda_status != cudaSuccess)
        {
            printf("CUDA SYNCRONIZE ERROR!\n");
            cudaFree(dev_weight);
            cudaFree(dev_firstSeq);
            return EXIT_FAILURE;
        }

        maxScore[i] = (int*)malloc(sizeof(int)*size);
        if(!maxScore[i])
            return EXIT_FAILURE;

        cuda_status = cudaMemcpy(maxScore[i], dev_max_score, sizeof(int)*size, cudaMemcpyDeviceToHost);
        if(cuda_status != cudaSuccess)
        {
            printf("CUDA SCORE1 ERROR!\n");
            cudaFree(dev_weight);
            cudaFree(dev_firstSeq);
            return EXIT_FAILURE;
        }

        maxOffset[i] = (int*)malloc(sizeof(int)*size);
        if(!maxOffset[i])
            return EXIT_FAILURE;
        
        cuda_status = cudaMemcpy(maxOffset[i], dev_max_offset, sizeof(int)*size, cudaMemcpyDeviceToHost);
        if(cuda_status != cudaSuccess)
        {
            printf("CUDA OFFSET1 ERROR!\n");
            cudaFree(dev_weight);
            cudaFree(dev_firstSeq);
            return EXIT_FAILURE;
        }

        maxMutant[i] = (int*)malloc(sizeof(int)*size);
        if(!maxMutant[i])
            return EXIT_FAILURE;

        cuda_status = cudaMemcpy(maxMutant[i], dev_max_mutant, sizeof(int)*size, cudaMemcpyDeviceToHost);
        if(cuda_status != cudaSuccess)
        {
            printf("CUDA MUTANT1 ERROR!\n");
            cudaFree(dev_weight);
            cudaFree(dev_firstSeq);
            return EXIT_FAILURE;
        }

        sizeMat[i] = size;

    }

    cudaFree(dev_max_score);
    cudaFree(dev_max_offset);
    cudaFree(dev_max_mutant);
    cudaFree(dev_firstSeq);
    cudaFree(dev_weight);
    cudaFree(sequences);

    return EXIT_SUCCESS;
}

