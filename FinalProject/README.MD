#Full code to replicate results
[Project_Rann.xlsx](https://github.com/bradley-rann/DScourseS23/files/11447750/Project_Rann.xlsx)

#R code given dataset

#import data
library(readxl)
data <- read_excel("C:/Users/radle/Downloads/Project_Rann.xlsx")
View(data)

#Format properly from CSV

transposed_data <- as.data.frame(transposed_data)
transposed_data <- transposed_data[-1, ]


# Rename the columns with the variables you provided
colnames(transposed_data) <- c("Total National Health Expenditures", "Educ", "Total Hospital Expenditures", "Out of pocket", "Obes", "Age", "Smok")

# Print the updated data
print(transposed_data)

#impute values

# Load required libraries
library(zoo)

# Perform linear interpolation for "Obes" variable
transposed_data$Obes <- na.approx(transposed_data$Obes, na.rm = FALSE)

# Perform linear interpolation for "Smok" variable
transposed_data$Smok <- na.approx(transposed_data$Smok, na.rm = FALSE)

# Perform linear interpolation for "Educ" variable
transposed_data$Educ <- na.approx(transposed_data$Educ, na.rm = FALSE)

# Print the updated data
print(transposed_data)


#linear regression model
subset_data <- transposed_data[11:59, ]
# Convert variables to numeric
subset_data$Obes <- as.numeric(subset_data$Obes)
subset_data$Age <- as.numeric(subset_data$Age)
subset_data$Educ <- as.numeric(subset_data$Educ)
subset_data$Smok <- as.numeric(subset_data$Smok)



model1 <- lm(`Total National Health Expenditures` ~ Obes + Age + Educ + Smok, data = subset_data)
model2 <- lm(`Total Hospital Expenditures` ~ Obes + Age + Educ + Smok, data = subset_data)
model3 <- lm(`Out of pocket` ~ Obes + Age + Educ + Smok, data = subset_data)

summary(model1)

#print to latex
library(xtable)

table1 <- xtable(model1)
table2 <- xtable(model2)
table3 <- xtable(model3)

print(table1, type = "latex")
print(table2, type = "latex")
print(table3, type = "latex")


#Plots of data
library(ggplot2)

subset_data$predicted <- predict(model1)
pdf("plot.pdf", width = 8, height = 6)
ggplot(subset_data, aes(x = Obes, y = `Total National Health Expenditures`)) +
  geom_point() +
  geom_line(aes(y = predicted), color = "red") +
  labs(x = "Obesity in %", y = "Total National Health Expenditures") +
  ggtitle("Linear Regression Model: Total National Health Expenditures") +
  theme_minimal()
