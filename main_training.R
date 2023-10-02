#phase of training of Q-matrix
library("readxl")

#in this RL-project is used the epsilon greedy policy
#the value of epsilon is not constant but decrease at every iteration
source("get_epsilon.R")

#in this function every time is simulated the entire pharmacology treatment of 48 weeks for 213 patients
source("fun_treatment_train.R")

#reading of the population parameters, every subject is defined with pk/pd parameters
pop <- read.csv("popReinforcement.csv")

pop <- pop[,c(1:17)]
nsogg <- length(pop$id)
pop$PLTi <- pop$basei
pop$BM1i <- pop$kin/pop$kt
pop$BM2i <- pop$kin/pop$kt
pop$BM3i <- pop$kin/pop$kt
pop$Guti <- 0
pop$Centrali <- 0
pop$Peripherali <- 0
pop$SC1i <- 0
pop$SC2i <- 0

#reading the state and action matrix to composed the q-matrix
stati <-  read_excel("stati.xlsx")
stati$De<- as.numeric(stati$De)
nstati <- length(stati$PLT)

azioni <- read_excel("azioni.xlsx")
nazioni <- length(azioni)
azioni <- data.matrix(azioni)


#list of q-matrix(130x34) for each patient
#130 state and 34 action
qlist <- list()
for(i in 1:nsogg){
  qlist[[i]] <- matrix(0,nrow=nstati,ncol=nazioni)
}


#parameters of the algorithm
nEpisodi_train <- 20000
gamma <- 0.95 #discount factor
alpha <- 0.1 #learning rate 

#validation
freqValidation <- 25 #rate of validation
episode_validation <- floor(nEpisodi_train/freqValidation ) 
numero_validazione <- 1 #count number of validation


#Qbig is a big list, the training is repeat for 20.000 times, at every validation is update the Qbig list
#each item cointain 213 Q-matrix. For the test set 
#i will choose the better Q-matrix for each patient

qBig <- list()
for (i in 1:episode_validation){
  qBig[[i]]<- qlist
}

reward_validation <- matrix(0,ncol = nsogg,nrow = episode_validation)
flagV <- 0 #1 if is an a validation run
train <- list()

#faccio andare il "processo" n episodi in modo tale che la rete possa apprendere
startTime <- Sys.time()
for(j in 1:nEpisodi_train){
  
  flagV<- 0 
  epsilon <- get_epsilon(episode=j)
  if(j%%freqValidation==0){
    #phase of validation, exploitation 
    flagV <- 1
    qBig [[numero_validazione]] <- qlist #add the validation qlist at the Qbig
    
  }
  
  train <- fun_treatment_train(pop=pop,epsilon=epsilon,nsogg=nsogg,
                                    gamma=gamma,
                                    alpha=alpha,
                                    stati=stati,
                                    azioni=azioni,
                                    qlist=qlist,
                               numero_validazione=numero_validazione,
                               reward_validation=reward_validation,
                               flagV=flagV) 
  
  
  
  if(j%%freqValidation==0){
    reward_validation <- train$reward_validation
    numero_validazione <- numero_validazione +1
  }else{
    qlist <- train$qlist
  }
  
  #to save the workspace during the running
  if(j%%200==0){
    print(paste("episode number: ",j,sep=""))
    save(list=ls(),file=paste(getwd(),"/data_training_Qlearning.RData",sep=""))
  }
  
  
}

endTime <- Sys.time()
print(endTime-startTime)

