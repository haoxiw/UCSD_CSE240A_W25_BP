//========================================================//
//  predictor.c                                           //
//  Source file for the Branch Predictor                  //
//                                                        //
//  Implement the various branch predictors below as      //
//  described in the README                               //
//========================================================//
#include <stdio.h>
#include <math.h>
#include "predictor.h"

//
// TODO:Student Information
//
const char *studentName = "Haoxi Wu";
const char *studentID = "TODO";
const char *email = "haw125@ucsd.edu";

//------------------------------------//
//      Predictor Configuration       //
//------------------------------------//

const char *bpName[4] = {"Static", "Gshare","Tournament", "Custom"};

int ghistoryBits; 
int lhistoryBits; 
int pcIndexBits;  
int bpType;       
int verbose;

//------------------------------------//
//      Predictor Data Structures     //
//------------------------------------//

uint32_t gHistoryRegister;      
uint8_t *BHT;                   

uint16_t *LHT;              
uint8_t *LPT;               
uint8_t Prediction_local;   

uint32_t pc_mask = 0;
uint32_t lHistory_mask = 0;
uint32_t gHistory_mask = 0;

uint32_t GHR;                 
uint8_t *GPT;                  
uint8_t *CPT;                   
uint8_t Prediction_global;      
uint8_t Prediction_tournament;  


int8_t PerceptronTable[256][16];   
uint16_t GHR_perceptron;            
uint8_t Prediction_perceptron;     
int numWeights = 16;                
uint8_t threshold = 33;             
int y_out;                         

uint32_t compute_XOR(uint32_t A, uint32_t B) {
    uint32_t result = (A ^ B);
    return result;
}

uint8_t GSharePredictor(uint32_t pc, uint32_t gHistoryRegister, uint8_t *BHT) {
    uint32_t GShare_mask = 0;
    for(int i=0;i<ghistoryBits;i++)
        GShare_mask += (1 << i);
    uint32_t pc_masked = pc & GShare_mask;
    uint32_t gHistoryRegister_masked = gHistoryRegister & GShare_mask;
    uint32_t XOR_index = compute_XOR(pc_masked, gHistoryRegister_masked);
    uint32_t correct_index = XOR_index & GShare_mask;
    uint8_t Prediction = (*(BHT+correct_index) >> 1);
    return Prediction;
}

uint8_t TournamentPredictor(uint32_t pc) {

    uint32_t pc_masked = (pc & pc_mask);
    GHR = (GHR & gHistory_mask);
    uint32_t LHT_value = *(LHT + pc_masked) & lHistory_mask;
    uint8_t localPrediction = *(LPT + LHT_value) >> 1;
    uint8_t globalPrediction = *(GPT + GHR) >> 1;
    uint8_t choicePrediction = *(CPT + GHR) >> 1;
    uint8_t TournamentPrediction;
    if (choicePrediction == 1) {
        TournamentPrediction = globalPrediction;
    }
    else
    {
        TournamentPrediction = localPrediction;
    }
    return TournamentPrediction;
}

uint8_t PerceptronPredictor(uint32_t pc) {
    
    uint32_t pc_masked = (pc & pc_mask);
    GHR_perceptron = (GHR_perceptron & gHistory_mask);
    //Just to capture the global history
    pc_masked = (pc_masked ^ GHR_perceptron) & pc_mask;
    //Most Significant byte of the Vector register is the bias
    y_out = PerceptronTable[pc_masked][numWeights-1];
    //Computing the rest of the elements of the addition
    for(int i=0;i<numWeights-1;i++) {
        uint8_t GHR_i = (GHR_perceptron >> i) & 1;
        if (GHR_i == 1)
            y_out = y_out + PerceptronTable[pc_masked][i];
        else
            y_out = y_out - PerceptronTable[pc_masked][i];
    }
    Prediction_perceptron = (y_out >= 0) ? 1 : 0;

    return Prediction_perceptron;
}

void update_BHT(uint32_t pc, uint8_t outcome, uint8_t *BHT) {
    uint32_t pc_mask = 0;
    for(int i=0;i<ghistoryBits;i++)
        pc_mask += (1 << i);
    uint32_t pc_index = (pc & pc_mask);
    uint32_t xor_index = compute_XOR(pc_index,gHistoryRegister);
    xor_index = xor_index & pc_mask;

    uint32_t current_status = *(BHT + xor_index);
    switch(current_status) {
        case 0:
            if (outcome == 0)
                *(BHT+xor_index) = 0;
            else if (outcome == 1)
                *(BHT+xor_index) = 1;
            break;
        case 1:
            if (outcome == 0)
                *(BHT+xor_index) = 0;
            else if (outcome == 1)
                *(BHT+xor_index) = 2;
            break;
        case 2:
            if (outcome == 0)
                *(BHT+xor_index) = 1;
            else if (outcome == 1)
                *(BHT+xor_index) = 3;
            break;
        case 3:
            if (outcome == 0)
                *(BHT+xor_index) = 2;
            else if (outcome == 1)
                *(BHT+xor_index) = 3;
            break;
        default:
            break;
    }
}

void update_CPT(uint8_t outcome_local, uint8_t outcome_global, uint8_t outcome, uint32_t index, uint8_t *CPT) {
    uint32_t index_mask = 0;
    
    for(int i=0;i<ghistoryBits;i++)
        index_mask += (1 << i);

    uint32_t actual_index = index & index_mask;
    
    uint8_t current_status = *(CPT + actual_index);
    switch(current_status) {
        case 0:
            {
                if (outcome == outcome_global && outcome != outcome_local)
                    *(CPT + actual_index) = 1;
                else
                    *(CPT + actual_index) = 0;
                break;
            }
        case 1:
            {
                if (outcome == outcome_global && outcome != outcome_local)
                    *(CPT + actual_index) = 2;
                else if (outcome == outcome_local && outcome != outcome_global)
                    *(CPT + actual_index) = 0;
                else
                    *(CPT + actual_index) = 1;
                break;
            }
        case 2:
            {
                if (outcome == outcome_global && outcome != outcome_local)
                    *(CPT + actual_index) = 3;
                else if (outcome == outcome_local && outcome != outcome_global)
                    *(CPT + actual_index) = 1;
                else
                    *(CPT + actual_index) = 2;
                break;
            }
        case 3:
            {
                if (outcome == outcome_local && outcome != outcome_global)
                    *(CPT + actual_index) = 2;
                else
                    *(CPT + actual_index) = 3;
                break;
            }
        default:
                break;
    }
}

void
train_predictor(uint32_t pc, uint8_t outcome)
{
  if (bpType == GSHARE) {
      update_BHT(pc, outcome, BHT);
      gHistoryRegister = gHistoryRegister << 1; 
      gHistoryRegister += outcome;
  }
  else if (bpType == TOURNAMENT) {
      uint32_t pc_index = pc & pc_mask;
      GHR = GHR & gHistory_mask;
    
      uint32_t LHT_value = *(LHT + pc_index) & lHistory_mask;
      uint8_t globalPrediction = *(GPT + GHR);
      uint8_t localPrediction = *(LPT + LHT_value);
      
      update_CPT((localPrediction >> 1),(globalPrediction >> 1), outcome, GHR,CPT);      

      switch(localPrediction) {
          case 0:
              {
                  if (outcome == 0)
                      *(LPT + LHT_value) = 0;
                  else
                      *(LPT + LHT_value) = 1;
                  break;
              }
          case 1:
              {
                  if (outcome == 0)
                      *(LPT + LHT_value) = 0;
                  else
                      *(LPT + LHT_value) = 2;
                  break;
              }
          case 2:
              {
                  if (outcome == 0)
                      *(LPT + LHT_value) = 1;
                  else
                      *(LPT + LHT_value) = 3;
                  break;
              }
          case 3:
              {
                  if (outcome == 0)
                      *(LPT + LHT_value) = 2;
                  else
                      *(LPT + LHT_value) = 3;
                  break;
              }
          default:
              break;
      }

      switch(globalPrediction) {
          case 0:
              {
                  if (outcome == 0)
                      *(GPT + GHR) = 0;
                  else
                      *(GPT + GHR) = 1;
                  break;
              }
          case 1:
              {
                  if (outcome == 0)
                      *(GPT + GHR) = 0;
                  else
                      *(GPT + GHR) = 2;
                  break;
              }
          case 2:
              {
                  if (outcome == 0)
                      *(GPT + GHR) = 1;
                  else
                      *(GPT + GHR) = 3;
                  break;
              }
          case 3:
              {
                  if (outcome == 0)
                      *(GPT + GHR) = 2;
                  else
                      *(GPT + GHR) = 3;
                  break;
              }
          default:
              break;
      }

      GHR = GHR << 1;
      GHR += outcome;
      GHR = GHR & gHistory_mask;
      
      *(LHT + pc_index) = (*(LHT + pc_index) << 1) + outcome;
      *(LHT + pc_index) = *(LHT + pc_index) & lHistory_mask;
  }
  else if (bpType == CUSTOM) {
      uint32_t pc_masked = (pc & pc_mask);
      pc_masked = (pc_masked ^ GHR_perceptron) & pc_mask;
      if ((Prediction_perceptron != outcome) || abs(y_out) <= threshold) {
          if (outcome == 1) {
              PerceptronTable[pc_masked][numWeights - 1] += 1;
          } else {
              PerceptronTable[pc_masked][numWeights - 1] -= 1;
          }

          for(int i=0;i<numWeights-1;i++) {
              uint8_t GHR_i = (GHR_perceptron >> i) & 1;
              uint8_t learning_rate = 1;
              if (i >= 7) {
                  if (outcome == GHR_i) {
                      PerceptronTable[pc_masked][i] += 1 * learning_rate;
                  } else {
                      PerceptronTable[pc_masked][i] -= 1 * learning_rate;
                  }
              } else {
                  if (outcome == GHR_i) {
                      PerceptronTable[pc_masked][i] += 1;
                  } else {
                      PerceptronTable[pc_masked][i] -= 1;
                  }
              }
          }
      }
      GHR_perceptron = GHR_perceptron << 1;
      GHR_perceptron += outcome;
      GHR_perceptron = GHR_perceptron & gHistory_mask;
  }
}


void
init_predictor()
{

    if (bpType == GSHARE) {
        gHistoryRegister = 0;
        int size_BHT = (1 << ghistoryBits);
        BHT = (uint8_t *) malloc(size_BHT * sizeof(uint8_t));
        for(int i=0;i<size_BHT;i++)
            *(BHT+i) = 1;
    }
    else if (bpType == TOURNAMENT) {
        //Declaring the masks
        pc_mask = (1 << (pcIndexBits+4)) - 1;
        lHistory_mask = (1 << lhistoryBits) - 1;
        gHistory_mask = (1 << ghistoryBits) - 1;

        GHR = 0;
        GHR = GHR & gHistory_mask;

        uint32_t size_LPT = (1 << lhistoryBits);
        uint32_t size_GPT = (1 << ghistoryBits);
        uint32_t size_LHT = (1 << pcIndexBits);

        LHT = (uint16_t *) malloc (size_LHT * sizeof(uint16_t));
        LPT = (uint8_t *) malloc (size_LPT * sizeof(uint8_t));
        GPT = (uint8_t *) malloc (size_GPT * sizeof(uint8_t));
        CPT = (uint8_t *) malloc (size_GPT * sizeof(uint8_t));

        for(int i=0;i<size_LHT;i++) {
            *(LHT + i) = 0;
            *(LHT + i) = *(LHT + i) & lHistory_mask;
        }

        for(int i=0;i<size_LPT;i++)
            *(LPT + i) = 1;

        for(int i=0;i<size_GPT;i++) {
            *(GPT + i) = 1;
            *(CPT + i) = 2;
        }
        
    }
    else if (bpType == CUSTOM) {
        //Declaring the bit-widths and masks
        ghistoryBits = numWeights;
        lhistoryBits = 10;
        pcIndexBits  = 7;

        //N - Size of the vector Register (Number of weights in the vector)
        uint8_t N = numWeights;
        
        pc_mask = (1 << pcIndexBits) - 1;
        lHistory_mask = (1 << lhistoryBits) - 1;
        gHistory_mask = (1 << ghistoryBits) - 1;
        
        GHR_perceptron = 0;
        GHR_perceptron = GHR_perceptron & gHistory_mask;

        uint32_t size_PerceptronTable = (1 << pcIndexBits);
        uint32_t size_VectorRegisters = numWeights;


        for(int i=0;i<size_PerceptronTable;i++) {
            for(int j=0;j<size_VectorRegisters;j++) {
                PerceptronTable[i][j] = -1;
            }
        }
    }
}


uint8_t
make_prediction(uint32_t pc)
{

  switch (bpType) {
    case STATIC:
      return TAKEN;
      break;
    case GSHARE:
      return GSharePredictor(pc, gHistoryRegister, BHT);
      break;
    case TOURNAMENT:
      return TournamentPredictor(pc);
      break;
    case CUSTOM:
      return PerceptronPredictor(pc);
      break;
    default:
      break;
  }

  return NOTTAKEN;
}

