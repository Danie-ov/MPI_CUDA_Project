#include <mpi.h>
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include "myProto.h"
#include <limits.h>

void printSeq(char** arr, int size)
{
   int i;
   for(i=0; i<size; i++)
   {
      printf("%s\n", arr[i]);
   }
}

void printArr(int* arr, int size)
{
   int i=0;
   for(i=0; i < size; i++)
      printf("%d ", arr[i]);
}

void printArrs(int* offsetArr, int* mutantArr, int size)
{
   int i;
   for(i=0; i < size; i++)
      printf("n = %d k = %d\n", offsetArr[i], mutantArr[i]);
}

int checkFirstGroup(char a, char b)
{
   const char *firstGroup[FIRST_GROUP_SIZE] = { "NDEQ", "NEQK", "STA", "MILV", "QHRK", "NHQK", "FYW", "HY", "MILF" };
   
   int i;
   for(i=0; i < FIRST_GROUP_SIZE; i++)
   {
      if(strchr(firstGroup[i], a) != NULL && strchr(firstGroup[i], b) != NULL)
            return 1;
   }
   return 0;
}

int checkSecondGroup(char a, char b)
{
   const char *secondGroup[SECOND_GROUP_SIZE] = { "SAG", "ATV", "CSA", "SGND", "STPA", "STNK", "NEQHRK", "NDEQHK", "SNDEQK", "HFY", "FVLIM" };

   int i;
   for(i=0; i < SECOND_GROUP_SIZE; i++)
   {
      if(strchr(secondGroup[i], a) != NULL && strchr(secondGroup[i], b) != NULL)
            return 1;
   }
   return 0;
}

void calculateAlignmentScore(int* weight, char** sequences,  char* firstSeq, int numOfSeq, int* allScores, int* allOffsets, int* allMutants)
{
   char a, b;
   int firstSeqLen = strlen(firstSeq), score, i;
#pragma omp parallel for private(i) private(score) private(a) private(b)
   for(i=0; i < numOfSeq; i++)
   {
      int nextSeqLen = strlen(sequences[i]), firstSeqLen = strlen(firstSeq);
      allScores[i] = INT_MIN;
      allOffsets[i] = 0;
      allMutants[i] = 1;
      for(int j=0; j < firstSeqLen - nextSeqLen; j++)
      {
         for(int k=1; k <= nextSeqLen; k++)
         {
            score = 0;
            for(int h=0; h <= nextSeqLen; h++)
            {
               a = firstSeq[h+j];
               if(h < k)
                  b = sequences[i][h];
                  
               else if(h > k)
                  b = sequences[i][h-1];
                  
               if(h != k)
               {
                  if(a == b)
                     score += weight[0];
                  
                  else if(checkFirstGroup(a, b))
                     score -= weight[1];
                  
                  else if(checkSecondGroup(a, b))
                     score -= weight[2];
                  else
                     score -= weight[3];
               }                  
            }
            if(score > allScores[i])
            {
               allScores[i] = score;
               allOffsets[i] = j;
               allMutants[i] = k;
            }
         }
      }
   }
}
 

int main(int argc, char *argv[]) {
    int size, rank, numOfSeq, i, j, k, h, offsetNum;
    char** sequences;
    int weight[4];
    char* firstSeq;
    char buffer[BUF_SIZE];
    MPI_Status  status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    
    char buff[BUFFER];
    int position = 0, firstSeqLen, workerSize, seqSize, score = 0, *maxScore, *maxOffset, *maxMutant, rem = 0;

    if (rank == 0) {
      fgets(buffer, BUF_SIZE, stdin);
      sscanf(buffer, "%d %d %d %d", &weight[0], &weight[1], &weight[2], &weight[3]);
      fgets(buffer, BUF_SIZE, stdin);
      buffer[strcspn(buffer, "\n")] = '\0';
      firstSeq = strdup(buffer);
      if(firstSeq == NULL)
      {
         printf("Error with input of first sequence\n");
         MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
      }
      fgets(buffer, BUF_SIZE, stdin);
      sscanf(buffer, "%d", &numOfSeq);
      sequences = (char**)malloc(sizeof(char*)*numOfSeq);
      if(!sequences)
      {
         printf("Error\n");
         MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
      }

      for(i=0; i<numOfSeq; i++)
      {
         fgets(buffer, BUF_SIZE, stdin);
         buffer[strcspn(buffer, "\n")] = '\0';
         sequences[i] = strdup(buffer);
      }
      firstSeqLen = strlen(firstSeq) + 1;
      workerSize = numOfSeq/size;

      maxScore = (int*)malloc(sizeof(int)*numOfSeq);
      if(!maxScore)
      {
         printf("Error\n");
         MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
      }

      maxOffset = (int*)calloc(numOfSeq, sizeof(int));
      if(!maxOffset)
      {
         printf("Error\n");
         MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
      }

      maxMutant = (int*)calloc(numOfSeq, sizeof(int));
      if(!maxMutant)
      {
         printf("Error\n");
         MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
      }

      int* pos = (int*)calloc(size-1, sizeof(int));
      if(!pos)
      {
         printf("Error\n");
         MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
      }

      
      rem = numOfSeq%size, i = 1, j = 0;
      while(i < size)
      {
         MPI_Pack(weight, 4 , MPI_INT, buff, BUFFER, &pos[i-1], MPI_COMM_WORLD);
         MPI_Pack(&firstSeqLen, 1, MPI_INT, buff, BUFFER, &pos[i-1], MPI_COMM_WORLD);
         MPI_Pack(firstSeq, firstSeqLen, MPI_CHAR, buff, BUFFER, &pos[i-1], MPI_COMM_WORLD);
         MPI_Pack(&workerSize, 1, MPI_INT, buff, BUFFER, &pos[i-1], MPI_COMM_WORLD);
         while(j < workerSize)
         {
            seqSize = workerSize * i + rem + j;
            int newSize =  strlen(sequences[seqSize]) + 1;
            MPI_Pack(&newSize, 1, MPI_INT, buff, BUFFER, &pos[i-1], MPI_COMM_WORLD);
            MPI_Pack(sequences[seqSize], newSize, MPI_CHAR, buff, BUFFER, &pos[i-1], MPI_COMM_WORLD);
            j++;
         }
         MPI_Send(buff, pos[i-1], MPI_PACKED, i ,0 ,MPI_COMM_WORLD);
         i++;
         j=0;
      }
      calculateAlignmentScore(weight, sequences, firstSeq, workerSize + rem, maxScore, maxOffset, maxMutant);
      for(i=1; i < size; i++)
       {
         offsetNum = workerSize*i + rem;
         MPI_Recv(maxScore + offsetNum, workerSize, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
         MPI_Recv(maxOffset + offsetNum, workerSize, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
         MPI_Recv(maxMutant + offsetNum, workerSize, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
       }
       printArrs(maxOffset, maxMutant, numOfSeq);
      
    }else{
      MPI_Recv(buff, BUFFER, MPI_PACKED, ROOT, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Unpack(buff, BUFFER, &position, weight, 4, MPI_INT, MPI_COMM_WORLD);
      MPI_Unpack(buff, BUFFER, &position, &firstSeqLen, 1, MPI_INT, MPI_COMM_WORLD);
      firstSeq = (char*)malloc(sizeof(char)*firstSeqLen);
      if(!firstSeq)
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); 
      MPI_Unpack(buff, BUFFER, &position, firstSeq, firstSeqLen, MPI_CHAR, MPI_COMM_WORLD);
      MPI_Unpack(buff, BUFFER, &position, &workerSize, 1 , MPI_INT, MPI_COMM_WORLD);
      sequences = (char**)malloc(sizeof(char*)*workerSize);
      if(!sequences)
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); 
      for(i=0; i < workerSize; i++)
      {
        MPI_Unpack( buff, BUFFER, &position, &seqSize, 1, MPI_INT, MPI_COMM_WORLD);
        sequences[i] = (char*)malloc(sizeof(char)*seqSize);
        if(!sequences[i])
          MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
      MPI_Unpack(buff, BUFFER, &position, sequences[i], seqSize, MPI_CHAR, MPI_COMM_WORLD);
      }
      
      maxScore = (int*)malloc(sizeof(int)*workerSize);
      if(!maxScore)
         return EXIT_FAILURE;

      maxOffset = (int*)calloc(workerSize, sizeof(int));
      if(!maxOffset)
         return EXIT_FAILURE;

      maxMutant = (int*)calloc(workerSize, sizeof(int));
      if(!maxMutant)
         return EXIT_FAILURE;

      if(rank%2 == 0)
      {
         calculateAlignmentScore(weight, sequences, firstSeq, workerSize, maxScore, maxOffset, maxMutant);
      }else{

         int** scores = (int**)malloc(sizeof(int*)*workerSize);
         if(!scores)
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);

         int** offsets = (int**)malloc(sizeof(int*)*workerSize);
         if(!offsets)
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);

         int** mutants = (int**)malloc(sizeof(int*)*workerSize);
         if(!mutants)
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);

         int* sizeMat = (int*)malloc(sizeof(int)*workerSize);
         if(!sizeMat)
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);

         if(computeOnGPU(weight, firstSeq, sequences, workerSize, scores, offsets, mutants, sizeMat) != EXIT_SUCCESS)
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);

         for(i=0; i < workerSize; i++)
         {
            maxScore[i] = scores[i][0];
            for(j=1; j < sizeMat[i]; j++)
            {
               if(scores[i][j] > maxScore[i])
               {
                  maxScore[i] = scores[i][j];
                  maxOffset[i] = offsets[i][j];
                  maxMutant[i] = mutants[i][j];
               }
            }
         }
      }
      
      MPI_Send(maxScore, workerSize, MPI_INT, ROOT , 0 ,MPI_COMM_WORLD);
      MPI_Send(maxOffset, workerSize, MPI_INT, ROOT , 0 ,MPI_COMM_WORLD);
      MPI_Send(maxMutant, workerSize, MPI_INT, ROOT , 0 ,MPI_COMM_WORLD);
   }
   
    
   MPI_Finalize();

    return 0;
}




