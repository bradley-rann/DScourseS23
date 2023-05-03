#import packages
library(sampleSelection)
library(tidyverse)
library(broom)
library(modelsummary)
library(gtExtras)


#import dataset
library(readr)
wages <- read_csv("ProblemSets/PS12/wages12.csv")

#Factorize variables 

college_factor <- as.factor(wages$college)
married_factor <- as.factor(wages$married)
union_factor <- as.factor(wages$union)

#fixing the data set

wages_factor <- data.frame(logwage = wages$logwage, hgc = wages$hgc, college = college_factor, exper = wages$exper, married = married_factor, kids = wages$kids, union = union_factor)


#summary table with no imputations
model <- lm(logwage ~ hgc + college + exper + married + kids + union, data = wages_factor)
summary_table <- modelsummary(model, output = "latex")
summary_table

#Complete imputation
model_cc <- lm(logwage ~ hgc + union + college + exper + (exper^2), data = wages, na.action = na.omit)

#Impute with mean
wages_mean <- wages_factor
wages_mean$logwage[is.na(wages_factor$logwage)] <- mean(wages_factor$logwage, na.rm = TRUE)
model_mean <- lm(logwage ~ hgc + union + college + exper + (exper^2), data = wages_mean)

#impute with sampleselector

wages_factor$valid <- !is.na(wages_factor$logwage)
wages_factor$logwage_recoded <- ifelse(is.na(wages_factor$logwage), 0, wages$logwage)

#Heckman Selection
model_heckit <- selection(selection = valid ~ hgc + union + college + exper + married + kids,
                          outcome = logwage_recoded ~ hgc + union + college + exper + (exper^2),
                          data = wages_factor, method = "2step")

#Combing Models


models <- list(model_cc, model_mean, model_heckit)

# Set the names of the models
names(models) <- c("Complete Case", "Impute with Mean", "Heckman Selection")

# Generate a table that shows the outputs separately
modelsummary(models, output = "latex")

#Probit model


# Define utility function and estimate probit model
utility <- with(wages, hgc + college + exper + married + kids)

model_probit <- glm(union ~ utility, family = binomial(link = "probit"), data = wages)

# Display results
summary(model_probit)
wages$predProbit <- predict(model_probit, newdata = wages, type = "response")


#Counterfactual

# counterfactual policy: mothers and wives not allowed to work in unions 
parms.newprefs <- model_probit
parms.newprefs$coefficients[c("married1","kids")] <- 0
wages$predprobitCfl <- predict(parms.newprefs, newdata = wages, type = "response")

view(wages)




