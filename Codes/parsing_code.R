##Quora Question-Pair Duplicate classification

#Working Directory
setwd("~/kaggle/quora_que/")

#Necessary library
library(data.table)
library(coreNLP)
library(pbapply)

#Load data

master_train <- fread("Data/train.csv",header = T,stringsAsFactors = F)
master_test <- fread("Data/test.csv",header = T,stringsAsFactors = F)

train <- rbind(master_train,master_test,fill =T)

libLoc = "~/R/x86_64-redhat-linux-gnu-library/3.3/coreNLP/extdata/stanford-corenlp-full-2015-12-09/"
initCoreNLP(
  mem = "1g",
  type = "english",
  libLoc,
  parameterFile = "~/emotional_connectivity/corenlp.properties"
)


q1_pos <- list()
q2_pos <- list()

start = 420001
iter <- nrow(train)
pb <- txtProgressBar(min = start, max = iter, style = 3)

t <- Sys.time()

options(warn = -1)

for(j in seq(start,iter,by = 5000)) {
  for (i in 1:5000) {
    q1_pos[[i]] <- coreNLP::annotateString(text = as.character(train$question1[j+i]))
    q2_pos[[i]] <- coreNLP::annotateString(text = as.character(train$question2[j+i]))
    setTxtProgressBar(pb, j+i)
  }
  cat(sprintf(paste("\nComplete = %0.2f ; Elapsed Time : ", difftime(Sys.time(), t,units = "hour"), "hours"), j * 100 / iter))
  
  saveRDS(q1_pos, paste0("RDSfiles/q1parsed_chunkt",j,".RDS"))
  saveRDS(q2_pos, paste0("RDSfiles/q2parsed_chunkt",j,".RDS"))
  
  q1_pos <- list()
  q2_pos <- list()
}
options(warn = 0)

q1_pos <- readRDS("q1parsed.RDS")
q2_pos <- readRDS("q2parsed.RDS")

for(j in seq(start,iter,by = 5000)){
  
  fname1 <- paste0("RDSfiles/q1parsed_chunkt",j,".RDS")
  fname2 <- paste0("RDSfiles/q2parsed_chunkt",j,".RDS")
  temp <- readRDS(fname1)
  q1_pos <- append(q1_pos,temp)
  temp <- readRDS(fname2)
  q2_pos <- append(q2_pos,temp)
  
  cat(j)
}

