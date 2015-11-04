library(plyr)
library(contrast)
library(reshape2)

# load the experiment
experiments  <- read.csv('data/poster_node_distance.csv')

# keyword is better than random
t.test(experiments$avg_random, experiments$avg_node_distance_kw, paired = T)

# scholarfy is better than keyword
t.test(experiments$avg_node_distance_kw, experiments$avg_node_distance , paired = T)

# keyword improves with votes
summary(lm(avg_node_distance_kw ~ number_recommend, experiments))

# scholarfy improves with votes
summary(lm(avg_node_distance ~ number_recommend, experiments))


# melt experiment data
melted_experiment <- melt(experiments, 
                          id.vars = c('number_recommend'), 
                          measure.vars = c('avg_node_distance', 'avg_node_distance_kw'), 
                          variable.name = 'method', 
                          value.name = 'distance')

melted_experiment$method <- factor(melted_experiment$method)

# keyword reduces error significantly less than scholarfy
summary(lm(distance ~ 1 + number_recommend + method:number_recommend, melted_experiment))
