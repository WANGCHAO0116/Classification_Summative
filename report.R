install.packages("readr")
install.packages("skimr")
install.packages("mlr3verse")
install.packages("ggforce")
install.packages("GGally")
install.packages("tidyverse")
install.packages("precrec")
install.packages("ranger")
install.packages("xgboost")
install.packages("DataExplorer")
install.packages("data.table")

library("tidyverse")
library("ggplot2")
library("GGally")
library("ggforce")
library("mlr3")
library("mlr3learners")
library("mlr3proba")
library("data.table")
library("mlr3verse")

set.seed(1)

hearts <- readr::read_csv("https://www.louisaslett.com/Courses/MISCADA/heart_failure.csv")

##skim the data set
skimr::skim(hearts)

hearts$fatal_mi <- as.factor(hearts$fatal_mi)

DataExplorer::plot_bar(hearts, ncol = 3)

##figure 1(A)
DataExplorer::plot_bar(hearts, by = "fatal_mi", ncol = 3)

DataExplorer::plot_histogram(hearts, ncol = 3)

##figure 1(B)
DataExplorer::plot_boxplot(hearts, by = "fatal_mi", ncol = 3)

##pairs plot
ggpairs(hearts %>% select(anaemia, diabetes, high_blood_pressure, sex, smoking))

##remove insensible variables
hearts <- hearts %>%
  select(-diabetes, -sex) 
skimr::skim(hearts)

##plot the parallel sets diagram
hearts.par <- hearts %>%
  select(fatal_mi, anaemia, high_blood_pressure, smoking) %>%
  group_by(fatal_mi, anaemia, high_blood_pressure, smoking) %>%
  summarize(value = n())

hearts.par$high_blood_pressure <- as.character(hearts.par$high_blood_pressure)
hearts.par$anaemia <- as.character(hearts.par$anaemia)
hearts.par$smoking <- as.character(hearts.par$smoking)

ggplot(hearts.par %>% gather_set_data(x = c(2:4)),
       aes(x = x, id = id, split = y, value = value)) +
  geom_parallel_sets(aes(fill = as.factor(fatal_mi)),
                     axis.width = 0.1,
                     alpha = 0.66) + 
  geom_parallel_sets_axes(axis.width = 0.15, fill = "lightgrey") + 
  geom_parallel_sets_labels(angle = 0) +
  coord_flip()

##define task
task_heart <- TaskClassif$new(id = "heart",
                              backend = na.omit(hearts),
                              target = "fatal_mi",
                              positive = "1")
task_heart

cv5 <- rsmp("cv", folds = 5)
cv5$instantiate(task_heart)

##define learners
lrn_baseline <- lrn("classif.featureless", predict_type = "prob")
lrn_lr <- lrn("classif.log_reg", predict_type = "prob")
lrn_lda <- lrn("classif.lda", predict_type = "prob")
lrn_cart <- lrn("classif.rpart", predict_type = "prob")
lrn_ranger  <- lrn("classif.ranger", predict_type = "prob")
lrn_xgboost <- lrn("classif.xgboost", predict_type = "prob")

##train learners on task
res <- benchmark(data.table(
  task       = list(task_heart),
  learner    = list(lrn_baseline,
                    lrn_lr,
                    lrn_lda,
                    lrn_cart,
                    lrn_ranger,
                    lrn_xgboost),
  resampling = list(cv5)
), store_models = TRUE)

##evaluate performance of all learners
res
res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))

trees <- res$resample_result(4)

tree1 <- trees$learners[[1]]

tree1_rpart <- tree1$model

plot(tree1_rpart, compress = TRUE, margin = 0.1)
text(tree1_rpart, use.n = TRUE, cex = 0.8)

lrn_cart_cv <- lrn("classif.rpart", predict_type = "prob", xval = 10)

res_cart_cv <- resample(task_heart, lrn_cart_cv, cv5, store_models = TRUE)

rpart::plotcp(res_cart_cv$learners[[5]]$model)

lrn_ranger$param_set

## Tune parameter using grid search
search_space = ps(
  num.trees = p_int(lower = 1, upper = 1000),
  mtry = p_int(lower = 1, upper = 10)
)

measure = msr("classif.acc")

evals20 = trm("evals", n_evals = 200)

instance = TuningInstanceSingleCrit$new(
  task = task_heart,
  learner = lrn_ranger,
  resampling = cv5,
  measure = measure,
  search_space = search_space,
  terminator = evals20
)

tuner = tnr("grid_search", resolution = 20)

tuner$optimize(instance)

instance$result_learner_param_vals

instance$result_y

##Fit a new random forest model with optimized parameters
lrn_new_ranger <- lrn("classif.ranger", predict_type = "prob", num.trees = 632, mtry = 2, id = 'new_ranger', importance = "permutation")

res_new_ranger <- resample(task_heart, lrn_new_ranger, cv5, store_models = TRUE)

res_new_ranger$aggregate(list(msr("classif.ce"),
                              msr("classif.acc"),
                              msr("classif.auc"),
                              msr("classif.fpr"),
                              msr("classif.fnr")))

##plot ROC curve for old & new random forest models
autoplot(res$resample_result(5), type = 'roc')
autoplot(res_new_ranger, type = 'roc')

