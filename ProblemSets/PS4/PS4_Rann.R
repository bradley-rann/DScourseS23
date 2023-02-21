library(tidyverse)
library(sparklyr)

#Set up connection to Spark
spark_install(version = "3.0.0")
sc <- spark_connect(master = "local")

#Create tibble
as_tibble(iris)
df <- copy_to(sc, df1)

#Copy tibble
sc <- spark_connect(master = "local")
* Using Spark: 3.0.0
> df1 <- as_tibble(iris)
> df <- copy_to(sc, df1)

#Select object and print
df %>% select(Sepal_Length,Species) %>% head %>% print

#Filter selected variable
df %>% filter(Sepal_Length>5.5) %>% head %>% print

#Combine above commands
df %>% select(Sepal_Length,Species) %>% filter(Sepal_Length>5.5) %>% head %>% print

#Use group by command
df2 <- df %>% group_by(Species) %>% summarize(mean = mean(Sepal_Length),
                                              count = n()) %>% head %>% print.
#Attempt sort command
df2 %>% arrange(Species) %>% head %>% print
