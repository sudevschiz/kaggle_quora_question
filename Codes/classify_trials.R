##Quora Question-Pair Duplicate classification

#Working Directory
setwd("~/kaggle/quora_que/")

#Necessary library
library(ranger)
library(data.table)
library(xgboost)
library(stringdist)
library(stringr)
library(caret)
library(e1071)
library(ROCR)
library(tm)
library(coreNLP)
library(foreach)
library(SnowballC)
library(pbapply)


doMC::registerDoMC(cores = 4)
#Load data

master_train <- fread("Data/train.csv",header = T,stringsAsFactors = F)
master_test <- fread("Data/test.csv",header = T,stringsAsFactors = F)

train <- master_train

#Data preparation on the entire training data before splitting
#train$question1 <- tolower(train$question1)
#train$question2 <- tolower(train$question2)





#Split training data 60:20:20

set.seed(23)
tr_vec <- sample(1:nrow(train),size = 0.6*nrow(train))
nontr_vec <- setdiff(1:nrow(train),tr_vec)
val_vec <- sample(nontr_vec,size = 0.5*length(tr_vec))
ts_vec <- setdiff(nontr_vec,val_vec)  

train$partition <- character(length = nrow(train))
train[tr_vec,]$partition <- "TRAIN" 
train[val_vec,]$partition <- "VAL"
train[ts_vec,]$partition <- "TEST"

train <- rbind(train,master_test,fill=T)
train[is.na(train$partition), "partition"] <- "FINAL_TEST"

#CLean

train$is_duplicate <- as.factor(train$is_duplicate)
levels(train$is_duplicate)


#Similar words
v <- strsplit(train$question1," ")
w <- strsplit(train$question2," ")
comWordsLength <- lapply(1:nrow(train), FUN = function(x){
  comWordsLength <- length(intersect(unlist(v[x]),unlist(w[x])))/length(unlist(v[x]))
}) %>% unlist()

train$comWordsLength <- comWordsLength


#Run benchmark count words model
logreg.model <-
  glm(is_duplicate ~ comWordsLength ,
      data = train[train$partition == "TRAIN", ],
      family = binomial(link = "logit"))

pred <- predict(logreg.model,newdata = train[train$partition  == "VAL",],type = "response")
perf_check <- function(pred,actual){
  
  rocr.pred <- prediction(as.numeric(pred[,2]),actual)
  
  
  # Recall-Precision curve             
  RP.perf <- performance(rocr.pred, "prec", "rec")
  
  plot (RP.perf)
  
  # ROC curve
  ROC.perf <- performance(rocr.pred, "tpr", "fpr")
  plot (ROC.perf)
  
  # ROC area under the curve
  auc.tmp <- performance(rocr.pred,"auc");
  auc <- as.numeric(auc.tmp@y.values)
  
  cat(paste(auc.tmp@y.name," :",auc.tmp@y.values))
  
  pred = sapply(pred,function(x) min(max(x, 1e-15), 1-1e-15)) 
  logLoss <- -1*mean(log(pred[model.matrix(~ actual + 0) - pred > 0]))
  
  cat(paste("\nLogLoss error :",logLoss,"\n"))
  
}


perf_check(matrix(c(1-pred,pred),ncol = 2),train[train$partition == "VAL",]$is_duplicate)

rf.model.bm <-
  ranger(
    is_duplicate ~ . ,
    data = train[train$partition == "TRAIN", 6:15],
    write.forest = T,
    importance = "impurity",
    probability = T,
    num.trees = 1001
  )
imp <- importance(rf.model.bm)

x <- names(imp)
plot(imp)
text(imp,labels = x)

rf.pred <- predict(object = rf.model.bm,data = train[train$partition == "VAL",6:15])



perf_check(rf.pred$predictions,train[train$partition == "VAL", ]$is_duplicate)
#Ll_error of 0.48 (Very bad)

#Remove correlatied features
##Only one string distance needed - jaccard
##OSA is highly correlated with num_words. Remove
##Remove length difference

rf.model1 <-
  ranger(
    formula = "is_duplicate ~ rep_qid1 + len_qid1 + len_qid2 + num_words_diff + jw_dist",
    train[train$partition == "TRAIN", 6:15],
    write.forest = T,
    importance = "impurity",
    probability = T,
    num.trees = 1001
  )

imp <- importance(rf.model1)
x <- names(imp)
plot(imp)
text(imp,labels = x)


rf.pred <- predict(object = rf.model1,data = train[train$partition == "VAL",6:15])

perf_check(rf.pred$predictions,train[train$partition == "VAL", ]$is_duplicate)


#This is giving a worse result than the benchmark. Taking away the features are reducing information.
#try adding more features?


train$cos_dist <- stringdist(train$question1,train$question2,method = "cosine")
train$jw_dist <- stringdist(train$question1,train$question2,method = "jw")
train$jaccard_dist <- stringdist(train$question1,train$question2,method = "jaccard")
train$osa_dist <- stringdist(train$question1,train$question2,method = "osa")


#number of characters
train$nchar1 <- nchar(train$question1)
train$nchar2 <- nchar(train$question2)

train$num_words_diff <- abs(str_count(train$question1,pattern = " ") - str_count(train$question2,pattern = " "))

rf.model2 <-
  ranger(
    is_duplicate ~ . ,
    data = train[train$partition == "TRAIN", 6:15],
    write.forest = T,
    #importance = "impurity",
    probability = T,
    num.trees = 501
  )

rf.pred <- predict(object = rf.model2,data = train[train$partition == "VAL",6:15])

perf_check(rf.pred$predictions,train[train$partition == "VAL", ]$is_duplicate)

#worse with stopwords. What is happening!!?

#More features 
#Number of caps
train$q1cap <- as.vector(unlist(lapply(train$question1,function(x){length(gregexpr(text = x,pattern = "[A-Z]")[[1]])})))
train$q2cap <-  as.vector(unlist(lapply(train$question2,function(x){length(gregexpr(text = x,pattern = "[A-Z]")[[1]])})))
train$capsequal <- (train$q1cap == train$q2cap)&(train$q1cap != 1)



#NLP features
#Obtained the POS tags of both the questions through a separate code.
#load them

q1_pos <- readRDS("q1parsed.RDS")
q2_pos <- readRDS("q2parsed.RDS")


# q1_pos <- pblapply(X = train$question1,FUN = function(x) {
#   coreNLP::annotateString(text = as.character(x))
# } )
# q2_pos <- pblapply(X = train$question2,FUN = function(x) {
#   coreNLP::annotateString(text = as.character(x))
# } )

train$NN <- NA
train$VB <- NA

tempq1NN <- lapply(q1_pos,function(x){ df <- as.data.frame(x); return(tolower(x[x$POS %in% c("NN","NNS","NNP"),"token"]))})
tempq2NN <- lapply(q2_pos,function(x){ df <- as.data.frame(x); return(tolower(x[x$POS %in% c("NN","NNS","NNP"),"token"]))})

tempq1VB <- lapply(q1_pos,function(x){ df <- as.data.frame(x); return(tolower(x[x$POS %in% c("VB","VBD","VBZ","VBG","VBN","VBP"),"token"] ))})
tempq2VB <- lapply(q2_pos,function(x){ df <- as.data.frame(x); return(tolower(x[x$POS %in% c("VB","VBD","VBZ","VBG","VBN","VBP"),"token"] ))})

train$NN <- length(intersect(tempq1NN,tempq2NN))/max(length(tempq1NN),length(tempq2NN))
train$VB <- length(intersect(tempq1VB,tempq2VB))/max(length(tempq1VB),length(tempq2VB))

train$NNANDVB <- train$NN & train$VB




rf.model3 <-
  ranger(
    formula = "is_duplicate ~ comWordsLength + cos_dist + jw_dist + jaccard_dist + osa_dist + num_words_diff + nchar1 +nchar2 + NN + VB + q1cap + q2cap+ capsequal + NNANDVB",
    data = train[train$partition == "TRAIN",],
    write.forest = T,
    importance = "permutation",
    probability = T,
    num.trees = 301,
    num.threads = 4
  )


imp <- importance(rf.model3)
x <- names(imp)
plot(imp)
text(imp,labels = x)



rf.pred <- predict(object = rf.model2,data = train[train$partition == "VAL",])

perf_check(rf.pred$predictions,train[train$partition == "VAL", ]$is_duplicate)



#Trying out xgboost model

target <- as.numeric(train$is_duplicate)
target <- ifelse(target == 1,0,1)

dtrain <-
  xgb.DMatrix(data.matrix(train[train$partition == "TRAIN", c(
    "comWordsLength",
    "cos_dist",
    "jw_dist",
    "jaccard_dist",
    "osa_dist",
    "num_words_diff",
    "nchar1",
    "nchar2",
    "NN",
    "VB",
    "q1cap",
    "q2cap",
    "capsequal",
    "NNANDVB"
  )]), label = target[train$partition == "TRAIN"], missing = NA)
  
  
  dval <- xgb.DMatrix(data.matrix(train[train$partition == "VAL", c(
    "comWordsLength",
    "cos_dist",
    "jw_dist",
    "jaccard_dist",
    "osa_dist",
    "num_words_diff",
    "nchar1",
    "nchar2",
    "NN",
    "VB",
    "q1cap",
    "q2cap",
    "capsequal",
    "NNANDVB"
  )]), label = target[train$partition == "VAL"], missing = NA)
  

watchlist <- list(train = dtrain,eval = dval)

param <- list(  objective           = "binary:logistic", 
                booster             = "gbtree",
                eval_metric         = "logloss",
                eta                 = 0.3, 
                max_depth           = 10, 
                subsample           = 0.8,
                colsample_bytree    = 0.8,
                min_child_weight    = 2,
                maximize            = FALSE
                )

xgb.model1 <-
  xgb.train(
    params = param,
    data = dtrain,
    nrounds = 500,
    verbose = 1,
    watchlist = watchlist,
    early_stopping_rounds = 10
  )

pred <- predict(xgb.model1,newdata = dval)


perf_check(matrix(c(1-pred,pred),ncol = 2),train[train$partition == "VAL",]$is_duplicate)

wdf <- write.csv(data.frame(cbind(train[train$partition == "VAL",],"PREDICTIONS" = pred)),file = "testout.csv")
